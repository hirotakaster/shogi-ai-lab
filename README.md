# Shogi AI Lab

ブラウザで対局できる将棋AIと、Google Colab の GPU で回す AlphaZero 方式の学習パイプラインです。

- ローカルは **依存ゼロ** で起動します（Python標準ライブラリのみ）
- 学習は **policy/value ネット + MCTS**（AlphaZero / dlshogi 方式）
- 学習は Colab、対局はローカル、という二段構成

## クイックスタート

```bash
python3 server.py
```

`http://127.0.0.1:8000` を開きます。PyTorch や学習済みモデルが無くても、
従来型αβエンジンで対局できます。

## Web UI

| 項目 | 内容 |
| --- | --- |
| 先手 / 後手 | 人間かAIを選びます |
| AIモデル | 従来型αβ、または学習モデル（MCTS） |
| 学習済みモデル | `data/*.pt` から選択。学習モデル選択時のみ有効 |
| 探索深さ (αβ) | 従来型エンジンの深さ |
| 探索数 (MCTS) | 学習モデルの1手あたりシミュレーション数 |
| 持ち時間 | 1手の秒読み。AI思考中もカウントは止まりません |

学習モデルが読み込めない場合は、理由を表示して従来型にフォールバックします。対局は中断しません。

右側の「AI同士の対局観戦」は自動対局の観戦機能です。学習は行いません。

## 学習（Google Colab）

`colab/train_shogi_ai.ipynb` を Colab で開きます。リポジトリを Google Drive に置いて、
ノートブックの `REPO` をそのパスに書き換えてから上から順に実行します。

ノートブックは次を行います。

1. GPU と vCPU 数の確認
2. `cshogi` の導入（C++実装。合法手生成が純Python比で約70倍速い）
3. **テスト実行による cshogi バックエンドの検証**
4. 速度実測から1ラウンドの局数を決定
5. 既存データでの事前学習
6. 自己対局 → 学習 → ゲーティング対戦のループ
7. 従来型エンジンとの対戦で強さを確認

### コマンドで直接回す場合

```bash
# 既存データで事前学習（推奨）
python -m shogi_ai.train_az --data data/initial_selfplay.jsonl \
    --epochs 20 --batch-size 256 --preset base --out data/policy_value.pt

# 自己対局 → 学習 → ゲーティングを10ラウンド
python -m shogi_ai.learn_loop --rounds 10 --games 60 --simulations 200 \
    --batch-size 32 --processes 8 --backend cshogi --preset base \
    --eval-games 20 --gate 0.55 --out-model data/policy_value.pt
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

`legal_moves()` の27,699回/秒には **SFEN/USI 橋渡しのコストが含まれます**。
cshogi 本体の合法手生成はこれより桁違いに速く、現在はアダプタ側が律速です。
ネイティブのマス番号を直接使えばさらに改善しますが、Mac で検証できないため
安全側に倒しています。

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

61テスト。cshogi 関連の11件は未導入環境（Apple Silicon など）ではスキップされます。

## 開発環境

- ローカル対局: Python 3.10以上、依存なし
- 学習: PyTorch + CUDA、`cshogi`
- `cshogi` は **Apple Silicon ではビルドできません**（setup.py が x86 の SSE/AVX
  フラグを強制するため）。Mac では純Pythonバックエンドを使います。

```bash
UV_CACHE_DIR=.uv-cache uv venv
UV_CACHE_DIR=.uv-cache uv pip install torch numpy   # 学習コードをローカルで触る場合のみ
```
