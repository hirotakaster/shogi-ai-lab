# AGENTS.md

## Project Goal

A locally runnable shogi AI with a Web interface for human play, and an
AlphaZero-style training pipeline (policy/value network + MCTS) that runs on a
Google Colab GPU. Local play must stay dependency-free; training is a Colab job.

## Architecture

- `server.py`: dependency-free HTTP JSON API and static file server.
- `shogi_ai/core.py`: rules, board representation, SFEN in and out, legal move generation, make-by-cloning.
- `shogi_ai/encoding.py`: the 85 feature planes and the 2187 policy indices. Shared by every backend.
- `shogi_ai/backends.py`: `PurePythonBackend` (no dependency) and `CshogiBackend` (fast).
- `shogi_ai/network.py`: the single definition of the ResNet policy/value net, plus self-describing checkpoints.
- `shogi_ai/evaluator.py`: batched inference on CUDA, MPS or CPU.
- `shogi_ai/mcts.py`: PUCT search with virtual-loss leaf batching.
- `shogi_ai/selfplay_az.py`: MCTS self-play producing training records.
- `shogi_ai/train_az.py`: joint policy and value training with a replay buffer.
- `shogi_ai/evaluate.py`: gating matches between generations, and against the heuristic engine.
- `shogi_ai/learn_loop.py`: the self-play / train / gate iteration driver.
- `shogi_ai/nn_model.py`: cached MCTS players for the Web server.
- `shogi_ai/engine.py`: the heuristic alpha-beta engine, kept as a baseline.
- `shogi_ai/self_play.py`: heuristic game generation for bootstrap data.
- `web/`: browser UI in plain HTML, CSS and JS.
- `colab/train_shogi_ai.ipynb`: the training notebook.

## Development Notes

- Keep the no-dependency launch path working: `python3 server.py` must start the
  Web UI with neither torch nor a checkpoint present, falling back to the
  heuristic engine and saying why in the UI.
- `encoding.py` is the contract between training and play. Normalisation to the
  side-to-move view happens there once; backends only supply raw absolute
  coordinates, so the two cannot drift apart. Change it and every existing
  checkpoint is invalid: bump `network.py`'s `CHECKPOINT_VERSION` so loading
  fails loudly instead of silently mis-evaluating.
- Checkpoints store their own architecture. Never match a `.pt` against a
  hard-coded class by hand; use `network.load_checkpoint`.
- cshogi moves its API between releases: `move_to_usi` is a **module** function,
  not a `Board` method. `CshogiBackend` resolves it at construction and accepts
  either spelling. Before trusting a new cshogi build run
  `python -m shogi_ai.backends`, which exercises every call the adapter makes
  and compares the colour mapping, legal moves, squares and hands against the
  pure-Python engine. `tests/test_cshogi_contract.py` pins that call structure
  with a fake module, so it runs without cshogi installed.
- `CshogiBackend` exchanges data with the encoder through SFEN and USI strings.
  That is deliberate: cshogi cannot be built on Apple Silicon, so this keeps
  the fast path running the same feature code as the reference path, and
  `tests/test_cshogi_backend.py` can pin the two together on Colab. Any native
  square-index fast path must keep that test green.
- MCTS values are stored from the point of view of the player to move at the
  node owning the edge, so a backup alternates sign. Terminal values are
  recorded on the node so a finished position is never selected from again.
- Shogi has no stalemate draw: a side with no legal move loses. `terminal_value()`
  returns `-1.0` for that, which is correct and is why a rook move that merely
  removes every legal reply is a winning move.
- Fourfold repetition is adjudicated as a draw in `PurePythonBackend`; cshogi
  reports perpetual check separately and that is honoured.
- The rules engine stays conservative: generated moves must not leave the
  mover's king in check.
- Falling loss does not prove added strength. A generation is promoted only
  after `shogi_ai.evaluate` shows it beating the current champion.
- Self-play is CPU-bound, not GPU-bound. `--processes` matters more than the
  size of the GPU; see the measured numbers in CLAUDE.md before tuning.
- Keep the AI turn clock running while an `/api/ai` request is in flight.
- AI time overrun must not become an automatic loss when a legal move is
  available. MCTS checks its deadline between batches, so it can overshoot by
  roughly one batch; leave headroom rather than tightening the check.
- Ignore `BrokenPipeError`/`ConnectionResetError` when a browser disconnects
  before an HTTP response is written.
- The browser only ever sees bare checkpoint filenames; `server.resolve_model`
  accepts a name only if it is in the current `data/` listing, so a crafted
  request cannot read outside the model directory.
- Stronger future versions should add:
  - left/right mirror augmentation (shogi is mirror-symmetric, so this is valid),
  - a native cshogi square mapping for feature building,
  - GPU batching across concurrent games,
  - mate search inside the tree,
  - entering-king declaration and pawn-drop mate,
  - the USI engine protocol.

## Verification Commands

```bash
python3 -m unittest discover -s tests   # 61 tests; 11 skip without cshogi
python3 server.py
python3 -m shogi_ai.self_play --games 2 --depth 2 --processes 2 --out data/smoke.jsonl
```

The neural pipeline, smallest useful settings:

```bash
python3 -m shogi_ai.selfplay_az --games 2 --simulations 24 --backend python \
    --preset small --out data/az_smoke.jsonl
python3 -m shogi_ai.train_az --data data/az_smoke.jsonl --epochs 2 \
    --preset small --workers 0 --out data/smoke.pt
python3 -m shogi_ai.evaluate --challenger data/smoke.pt --champion random \
    --games 2 --simulations 16 --backend python --preset small
```

On Colab, the cshogi agreement tests must actually run rather than skip:

```bash
pip install cshogi
python -m unittest tests.test_cshogi_backend -v
```

The JS has no test runner; syntax-check it with whatever is available:

```bash
node --check web/app.js
# or, on macOS without node:
/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc \
    -e "new Function(readFile('web/app.js')); print('ok')"
```
