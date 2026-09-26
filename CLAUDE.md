# CLAUDE.md

## 開発内容

Web画面で人間と対局でき、Google Colab の GPU で AlphaZero 方式の強化学習を回せる将棋AIです。

ローカルは依存ゼロで起動できる状態を保ち、学習は Colab で行う二段構成です。

## アーキテクチャ

学習と対局は policy/value ネット + MCTS（AlphaZero / dlshogi 方式）です。

```text
shogi_ai/
  core.py         将棋ルール、盤表現、SFEN入出力、合法手生成
  encoding.py     特徴量85プレーンと方策2187インデックス（全バックエンド共通）
  backends.py     PurePythonBackend（依存ゼロ）と CshogiBackend（高速）
  network.py      ResNet policy/value ネット唯一の定義とチェックポイント入出力
  evaluator.py    バッチ推論ラッパー（CUDA / MPS / CPU）
  mcts.py         PUCT探索、virtual lossによる葉のバッチ収集
  selfplay_az.py  MCTS自己対局 → 学習レコード生成
  train_az.py     policy + value 同時学習、リプレイバッファ
  evaluate.py     世代間のゲーティング対戦、従来型エンジンとの比較
  learn_loop.py   自己対局 → 学習 → ゲーティングの反復ドライバ
  nn_model.py     Web対局用のMCTSプレイヤ（チェックポイントをキャッシュ）
  engine.py       従来型αβ探索（ベースラインとして維持）
  self_play.py    従来型エンジンによる初期データ生成
colab/
  train_shogi_ai.ipynb  Colab学習ノートブック
```

## 表現の仕様

ここが全体の土台なので、変更するときは `tests/test_encoding.py` を必ず確認します。

- **入力は85プレーン**: 自駒14 + 敵駒14 + 自分の持ち駒28 + 相手の持ち駒28 + 王手1
- **持ち駒はサーモメータ符号化**: 歩は8枚、香桂銀金は4枚、角飛は2枚で飽和
- **常に手番側視点**: 後手番の局面は盤を180度回転し、自駒と敵駒を入れ替える
- **方策は2187出力**: 27プレーン（10方向 + 成り10方向 + 持ち駒7種）× 81マス
- 正規化は `encoding.py` の1箇所だけで行い、両バックエンドが同じコードを通ります

`encode_planes` と `policy_index` を変えると既存チェックポイントは無効になります。
`network.py` の `CHECKPOINT_VERSION` を上げてください。読み込み時に検出して明確に失敗します。

## バックエンドの使い分け

| | PurePythonBackend | CshogiBackend |
| --- | --- | --- |
| 依存 | なし | `pip install cshogi` |
| 合法手生成 | 約1,100回/秒 | 3桁速い |
| 用途 | ローカルWeb対局、テスト | Colabでの自己対局 |
| Apple Silicon | 動く | **ビルド不可**（setup.pyがx86のSSE/AVXフラグを強制） |

cshogi は Mac で検証できないため、`CshogiBackend` は **SFEN と USI を共通の橋渡しに使い、
特徴量は純Python実装とまったく同じコードで作ります**。これにより
`tests/test_cshogi_backend.py` が Colab で両者の一致を検証できます。

cshogi は版によって API の置き場所が変わります（`move_to_usi` は **モジュール関数** で、
`Board` のメソッドではありません）。`CshogiBackend` は生成時に解決し、どちらの形でも
受け付けます。新しい cshogi を使う前に次を実行してください。

```bash
python -m shogi_ai.backends
```

依存している呼び出しを全部検査し、**手番の色・合法手・盤面・持ち駒**を純Python実装と
照合します。色が反転すると学習データの符号が全部狂うため、ここを必ず通してください。
`tests/test_cshogi_contract.py` は偽モジュールで呼び出し構造を固定しており、
cshogi 未導入環境でも走ります。

## 実行方法

```bash
python3 server.py
```

起動後 `http://127.0.0.1:8000` を開きます。PyTorch と `data/*.pt` が無い場合、
UIは自動的に従来型エンジンのみになります（「学習モデル」が理由付きで無効化されます）。

## AIモデルの使い分け

Web UIの「AIモデル」は2種類です。

- **従来型 (Heuristics)**: 駒価値・位置評価とαβ探索。学習済みモデル不要、CPU動作。
- **学習モデル (Neural Net)**: `data/*.pt` を読み込み、MCTSで着手を選びます。CUDAがあれば使います。

「学習済みモデル」select box は使用する `.pt` を選ぶ項目で、学習モデル選択時のみ有効です。
「探索数 (MCTS)」も学習モデル選択時のみ有効で、「探索深さ (αβ)」は従来型用です。

モデルが読めない場合は従来型にフォールバックし、理由をUIに表示します。対局は止まりません。

右側の「AI同士の対局観戦」は、選択中のエンジンで自動対局させる観戦機能です。
**学習そのものは行いません**（学習はColabです）。

## Colabでの学習

`colab/train_shogi_ai.ipynb` を開き、リポジトリを Drive に置いて実行します。

```bash
# 既存データで事前学習（推奨、ランダム初期化より立ち上がりが速い）
python -m shogi_ai.train_az --data data/initial_selfplay.jsonl \
    --epochs 20 --batch-size 256 --preset base --out data/policy_value.pt

# 自己対局 → 学習 → ゲーティングを反復
python -m shogi_ai.learn_loop --rounds 10 --games 60 --simulations 200 \
    --batch-size 32 --processes 8 --backend cshogi --preset base \
    --eval-games 20 --gate 0.55 --out-model data/policy_value.pt
```

`learn_loop` は各ラウンドをサブプロセスで実行し、**ゲーティング対戦に勝ったモデルだけを昇格**
させます。成果物は `data/learn/` に残るため、Colabが切断されても同じコマンドで再開します。

### ネットワーク規模

`--preset` で選びます。チェックポイントに構成が埋め込まれ、読み込み時に自動復元されます。

| preset | 構成 | パラメータ |
| --- | --- | --- |
| `small` | 6ブロック × 96ch | 1.74M |
| `base` | 10ブロック × 192ch | 7.47M |
| `large` | 15ブロック × 256ch | 18.59M |

### 速度の実態

**自己対局のボトルネックは GPU ではなく CPU です。** 合法手生成と Python の
MCTS ツリー操作が律速するため、GPU を大きくしても自己対局の生成量はほとんど変わりません。
`--processes` を vCPU 数に合わせることが最も効きます。

純Pythonバックエンドの実測値（Apple M系, 200 sims/手）:

| 測定 | 値 |
| --- | --- |
| `legal_moves()` | 1,089回/秒（0.92ms） |
| MCTS（NN評価なし） | 約530 sims/秒 |
| MCTS（base / MPS） | 約258 sims/秒 |

この速度では1局あたり約2分かかり、1,000局で33時間になります。
これが Colab で cshogi を使う理由です。

## Web対局のタイマー仕様

- 人間とAIのどちらの手番でも、現在手番の秒読みカウンターを表示します。
- AIが思考中も `web/app.js` のタイマーは停止しません。
- UIで指定した時間をAI探索の上限として使い、通信遅延を避けるため約0.25秒の余裕を確保します。
- MCTSの時間制限はバッチ境界で判定するため、1バッチ分だけ超過しえます。
- AI探索が上限を超えても、合法な着手が得られていればその手を指します。時間超過を即敗北にはしません。
- 人間の時間切れは従来どおりWeb UIで判定します。

## テスト

```bash
python3 -m unittest discover -s tests
```

80テスト。`tests/test_cshogi_backend.py` の11件は cshogi 未導入環境ではスキップされます。
Colabでは必ず実行し、スキップされていないことを確認してください。

## 今後の強化ポイント

- 左右反転によるデータ拡張（将棋は左右対称なので有効）
- cshogi のマス番号を直接使う特徴量生成（現状はSFEN経由で安全側に倒している）
- 複数対局をまたぐGPU推論バッチング（現状は1局内のvirtual lossバッチのみ）
- 詰み探索の探索への統合
- 入玉宣言と打ち歩詰めの厳密対応
- USIプロトコル対応
