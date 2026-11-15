# Shogi AI Lab

ブラウザで対局できる将棋AIと、Google Colab の GPU で回す AlphaZero 方式の学習パイプラインです。

- ローカルは **依存ゼロ** で起動します（Python標準ライブラリのみ）
- 学習は **policy/value ネット + MCTS**（AlphaZero / dlshogi 方式）
- 学習は Colab、対局はローカル、という二段構成

## クイックスタート

```bash
git clone https://github.com/hirotakaster/shogi-ai-lab.git
cd shogi-ai-lab
uv sync
uv run python server.py
```

`http://127.0.0.1:8000` を開きます。`uv sync` はプロジェクトの仮想環境を用意します。
通常の対局に追加パッケージは不要で、PyTorch や学習済みモデルが無くても従来型αβエンジンで対局できます。

## Web UI

| 項目 | 内容 |
| --- | --- |
| 先手 / 後手 | 人間かAIを選びます |
| AIモデル | 従来型αβ、または学習モデル（NN + MCTS） |
| 学習済みモデル | `data/*.pt` から選択。学習モデル選択時のみ有効 |
| 探索深さ (αβ) | 従来型エンジンの深さ |
| 探索数 (MCTS) | 学習モデルの1手あたりシミュレーション数（初期値: 2000） |
| 秒読み音 | チェックを外すと通知音を消音できます（初期状態はオン） |
| 持ち時間 | 1手の秒読み。AI思考中もカウントは止まりません。音がオンの場合、人間の手番は残り9秒から毎秒鳴ります |

学習モデルが読み込めない場合は、理由を表示して従来型にフォールバックします。対局は中断しません。
実際の探索数は持ち時間によって初期値より少なくなる場合があります。

右側の「AI同士の対局観戦」は自動対局の観戦機能です。学習は行いません。

## 学習（Google Colab）

`colab/train_github.ipynb` を Colab で開きます。ノートブックは GitHub の最新の
`mochi/batch-*` ブランチを clone して使うので、リポジトリを Drive に置く必要は
ありません。Google Drive は成果物（チェックポイントと `data/learn/`）の保存先として
だけマウントします（`shogi-ai-lab-artifacts/`）。

ノートブックは次を行います。

1. GPU と vCPU 数の確認
2. `cshogi` の導入（C++実装。合法手生成が純Python比で約70倍速い）
3. **テスト実行による cshogi バックエンドの検証**
4. 速度実測から1ラウンドの局数を決定
5. 既存データでの事前学習
6. 自己対局 → 学習 → ゲーティング対戦のループ
7. 従来型エンジンとの対戦で強さを確認

継続学習（第2期以降）は、Drive のバックアップからチャンピオン
（`policy_value.pt`）だけを復元してリプレイバッファを新規に始めます。
ノートブックに復元専用セルがあり、事前学習のセルは飛ばします。学習ループは
ラウンドごとに成果物を Drive へ自動バックアップするので、ランタイムが
落ちても直近のラウンドから再開できます。

### コマンドで直接回す場合

```bash
# 既存データで事前学習（推奨）
python -m shogi_ai.train_az --data data/initial_selfplay.jsonl \
    --epochs 20 --batch-size 256 --preset base --out data/policy_value.pt

# 自己対局 → 学習 → ゲーティング（本番相当の設定。第2期の検証済み構成）
python -m shogi_ai.learn_loop --rounds 10 --games 400 --simulations 200 \
    --batch-size 32 --processes 10 --backend cshogi --preset base \
    --search-mode gumbel --gumbel-candidates 8 --playout-cap-fraction 0.25 \
    --round-lr 5e-4 --round-epochs 5 --latest-repeat 4 --grad-clip 1.0 \
    --round-max-steps 3600 \
    --eval-games 80 --eval-simulations 128 --gate 0.55 --eval-add-noise \
    --max-positions 400000 --out-model data/policy_value.pt
```

新しいモデルは、現行モデルに対して `--gate` 以上のスコアを取ったときだけ昇格します。
成果物は `data/learn/` に残るため、Colabが切断されても同じコマンドで再開します。

### ネットワーク規模

| preset | 構成 | パラメータ |
| --- | --- | --- |
| `small` | 6ブロック × 96ch | 1.74M |
| `base` | 10ブロック × 192ch | 7.47M |
| `large` | 15ブロック × 256ch | 18.59M |

チェックポイントに構成が埋め込まれるので、読み込み時にアーキテクチャは自動復元されます。

### 知っておくべき速度の話

**自己対局のボトルネックは GPU ではなく CPU です。** 合法手生成と Python の MCTS
ツリー操作が律速するため、GPU を大きくしても生成できる棋譜の量はほとんど増えません。
`--processes` を vCPU 数に合わせるのが最も効きます。

Colab (T4 / 8 vCPU) での実測値:

| 測定 | 純Python | cshogi |
| --- | --- | --- |
| `legal_moves()` | 393回/秒 | 27,699回/秒（**約70倍**） |

MCTS の実効速度（`base`、CUDA、1プロセス）:

| MCTS葉バッチ | sims/秒 |
| --- | --- |
| 8 | 516 |
| 16 | 1,758 |
| 32 | 2,314 |
| 64 | 3,140 |

上の `legal_moves()` の実測は SFEN/USI 文字列を経由していた頃の値です。
現在のアダプタは cshogi のネイティブなマス番号・指し手を直接読み書きし、
文字列の橋渡しはホットパスから外れています。プロファイルでは単一の
ホットスポットは残っておらず、コストは合法手生成とツリー操作に分散しています。

## 強さの測り方

損失が下がったことは強くなった証拠になりません。対戦で確認します。

```bash
# ランダム初期化ネットとの比較（下限確認）
python -m shogi_ai.evaluate --challenger data/policy_value.pt --champion random \
    --games 20 --simulations 200 --backend cshogi

# 従来型αβエンジンとの比較（--backend python のため低速）
python -m shogi_ai.evaluate --challenger data/policy_value.pt --champion heuristic \
    --games 10 --simulations 200 --heuristic-depth 3 --backend python
```

## 学習の記録

- **第1期（2026-11-14〜15、L4、40ラウンド）**: 22ラウンドで昇格し、最終
  チャンピオンは 41,892 steps。従来型エンジン（深さ3）との物差し対戦は
  0.200 → **0.338** に改善したが、まだ格下。次の目標は 0.500（物差しに勝つ）。
- **第2期（最初の10ラウンド完了）**: 第1期の最終チャンピオンから base のまま
  継続。T4 での A/B 検証で決めた設定（Gumbel 候補 8、ゲート 80 局）で最初の
  10ラウンドを回し、2ラウンド昇格してチャンピオンは 43,052 steps に
  （ラウンド8が 0.575 で昇格、9は 0.400、10は 0.494 で却下）。候補が王者と
  ほぼ互角止まりだったため、続くラウンドでは学習量を増やした
  （エポック 5、最新データの重み 4倍、ステップ上限 3600）。FPU は A/B で
  悪化したため不採用、シミュレーションは 200 のまま（400 に増やしても
  伸びなかった）。43,052 steps 時点の物差し対戦は未測定。

## 実装済み

- 合法手生成、成り、持ち駒打ち、王手回避、二歩、行き所のない駒
- 千日手の4回反復による引き分け判定
- 手番側視点に正規化した85プレーンの特徴量（持ち駒を含む）
- 2187出力の方策ヘッドと価値ヘッド
- virtual loss による葉のバッチ収集を備えた PUCT MCTS
- 投了、温度スケジュール、Dirichletノイズ付きの自己対局
- リプレイバッファと混合精度学習
- 学習時の左右反転データ拡張（方策インデックスも対称変換）
- 世代間ゲーティング対戦と Elo 換算表示
- Gumbel AlphaZero 探索（Sequential Halving による低シミュレーション向けのルート選択）
- playout cap randomisation（一部の手をフル探索、それ以外を低コストで指す）
- 探索木の再利用（指した手の部分木を次の手のルートに引き継ぐ）
- ゲーム横断のバッチ推論と、自己対局・ゲーティングの並列実行
- ゲーティングへの対局ごとのノイズ追加（先後だけで結果が決まる鏡像対戦の回避）
- ファインチューニング学習（低学習率・最新ラウンドの重み付け・固定ステップ数）
- ラウンドごとの Google Drive への自動バックアップ
- 自己記述型チェックポイント（非互換な旧形式は明確に拒否）
- 従来型αβエンジン（ベースライン）
- 依存ゼロのWebサーバと将棋盤UI

## 未対応

- 入玉宣言、打ち歩詰めの厳密判定
- USIプロトコル
- 置換表、詰み探索の探索統合

## テスト

```bash
python3 -m unittest discover -s tests
```

158テスト。cshogi 関連と PyTorch 必須のテストは、未導入環境（Apple Silicon など）ではスキップされます。

## 開発環境

- ローカル対局: Python 3.10以上、依存なし
- 学習: PyTorch + CUDA、`cshogi`
- `cshogi` は **Apple Silicon ではビルドできません**（setup.py が x86 の SSE/AVX
  フラグを強制するため）。Mac では純Pythonバックエンドを使います。

```bash
UV_CACHE_DIR=.uv-cache uv venv
UV_CACHE_DIR=.uv-cache uv pip install torch numpy   # 学習コードをローカルで触る場合のみ
```
