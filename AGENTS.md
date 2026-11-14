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
- `CshogiBackend` reads cshogi's native arrays where the build exposes
  them: `Board.pieces` / `Board.pieces_in_hand` for snapshots and planes,
  and the `move_to` / `move_from` / `move_is_drop` bit-field accessors
  for legal moves -- no SFEN or USI string is built or parsed on the hot
  path.  Builds without those accessors fall back to the SFEN/USI string
  exchange.  `verify_cshogi_api` cross-checks the native path against
  the string path, and `tests/test_cshogi_backend.py` pins both against
  the pure-Python engine, so a square-order or piece-table slip fails
  loudly instead of silently corrupting features.  cshogi squares are
  `(file - 1) * 9 + rank`; `_SQUARE_FROM_CSHOGI` in `backends.py` maps
  them to snapshot squares, and cshogi's piece-type order (P L N S B R G
  K ...) differs from `PIECE_ORDER` in gold's position -- both tables
  live next to the conversion, do not "simplify" them.
- Feature planes have two encoders that must agree: `encode_planes`
  (reference, snapshot in / planes out) and `encode_planes_array`
  (vectorised, NumPy squares in).  Backends with a `planes()` method are
  used directly by MCTS; everything else goes through `encode_planes`.
- MCTS values are stored from the point of view of the player to move at the
  node owning the edge, so a backup alternates sign. Terminal values are
  recorded on the node so a finished position is never selected from again.
- Self-play keeps the search tree between plies: `MCTS.advance` retains the
  played move's subtree as the next root. Root Dirichlet noise is always
  mixed from the raw network priors (`Node.raw_priors`), never from an
  already-noised distribution, so reuse cannot wash the priors out.
- A self-play game cut off by the move cap is not a draw: its records carry
  `capped: true`, and `train_az` uses the search's root value `q` as their
  value target instead of the forced-draw outcome.
- The search has two root modes (`MCTS(search_mode=...)`).  `puct` is the
  default.  `gumbel` is Gumbel AlphaZero for low simulation budgets: the
  top `gumbel_candidates` actions by `gumbel + log pi` are searched with
  Sequential Halving, and the root's policy target / move distribution is
  the improved policy `softmax(log pi + sigma(completed Q))` stored on
  `Node.improved_policy`, not the visit counts.  Gumbel mode never also
  applies Dirichlet noise (the Gumbel sample is the exploration), and
  with `add_noise=False` it is deterministic.  Below the root it is
  ordinary PUCT.
- Self-play supports playout-cap randomisation (`--playout-cap-fraction`):
  that fraction of plies is searched at the full budget and recorded;
  the rest are played at `--playout-cap-simulations` (default a quarter
  of the full budget) and are not recorded, because a capped search's
  policy target is too noisy to train on.  Capped plies also never count
  toward resignation.  Note the name overlap: a *record's* `capped: true`
  still means the move-cap adjudication above, not a low-cap ply.
- `MCTS.run(..., simulations=N)` overrides the budget for one call only;
  playout-cap randomisation is built on it.
- Self-play can run games concurrently in one process
  (`--parallel-games N`): each game owns its backend, search and RNG,
  and leaf evaluations are coalesced across games by
  `evaluator.BatchingEvaluator` (one background thread owns the real
  evaluator, so a CUDA context stays single-threaded).  This is the
  single-process path only; multiprocess workers are unchanged.
- Gating matches parallelise the same way self-play does
  (`evaluate --processes N`; `learn_loop --eval-processes` defaults to
  the self-play `--processes`).  A match game is seeded by
  (seed, game index) alone and the challenger always takes Black in
  even-indexed games, so which worker plays a game cannot change its
  outcome -- only the order results are tallied in.  Players are built
  per worker from their spec strings; a loaded network is never handed
  to another process.  With `--add-noise` off (the default) the gate is
  deterministic after the 8 temperature-1.0 opening plies, which the
  2026-11-13 production round 1 exposed as a flaw: the gate scored
  exactly 0.500 with the result alternating every game, i.e. the
  colour, not the strength, decided all 40 games between two
  near-identical nets.  `evaluate --add-noise` / `learn_loop
  --eval-add-noise` turns on per-game root noise (Gumbel sample in
  gumbel mode, Dirichlet noise in puct mode), drawn from the same
  per-game RNG, so a noisy match stays reproducible and
  parallel-invariant while near-identical nets can diverge;
  `--eval-opening-moves` widens the temperature-1.0 opening instead.
- Rounds that train *from* a champion use the finetuning regime
  (`--round-lr` / `--round-epochs` in `learn_loop`) when set, and
  `--latest-repeat N` oversamples the newest round's data in the
  replay buffer.  Both exist because of what the first Colab run
  measured (see the production recipe below): a 20-game round is
  ~1.3k positions and ~20 optimiser steps, and training that at the
  from-scratch rate of 2e-3 spiked the validation policy loss past 40
  mid-round -- the candidates came out as champion-plus-noise and
  both rounds were rejected at the gate (0.400, 0.250).
- The loop can mirror its committed state off the runtime
  (`learn_loop --backup-dir DIR`): at startup and after every round it
  copies the champion and the round's artefacts into DIR (atomically,
  history last -- it is the commit marker), in the same layout as the
  notebook's manual save cell, so the same restore procedure applies.
  This exists because a Colab runtime's disk is ephemeral: without it
  the manual save cell was the only copy, and a dead session lost every
  round since it last ran.  The notebook passes the Drive artefacts
  folder.  Backup failures warn and continue -- a flaky Drive mount
  must not kill a long run.
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
  - mate search inside the tree,
  - entering-king declaration and pawn-drop mate,
  - the USI engine protocol.

## Production-Scale Recipe (Colab)

Measured on the first full run (Batch 4 code, A100-80GB, 12 vCPUs):
self-play at 200 simulations with Gumbel + cap 0.25 ran at ~960
games/hour across 12 processes and recorded ~65 positions per game.
A 20-game round is therefore only ~1.3k positions -- two orders of
magnitude below what a round needs.  Budget from positions, not games:
**at least ~25k recorded positions per round**, i.e. 400+ games.

The planned production GPU is an **L4 (24 GB)**, which is ample for
this project: the base network is a few million parameters, training
at batch 256 with AMP stays far below 24 GB, and the replay buffer
lives on disk / host RAM, not in VRAM.  What changes versus the A100
measurements is speed, not feasibility: expect training steps to run
roughly 1.5-2x slower, while self-play is CPU-bound (cshogi + the
Python search tree), so its throughput tracks the *host vCPU count*,
not the GPU.  Check `nproc` on the L4 runtime first and set
`--processes` to vCPUs - 2; if the runtime has fewer vCPUs than the
A100 instance had, games/hour drops in proportion and the round size
should be trimmed before the simulations are.

```bash
# Bigger bootstrap (CPU-cheap; ~38k positions at 300 games)
python -m shogi_ai.self_play --games 300 --depth 2 --processes 10 \
    --out data/initial_selfplay.jsonl

# One command for the whole run; round 1 pretrains from the bootstrap
# inside the loop, later rounds finetune.  The loop is resumable, so
# this can span several Colab sessions -- just rerun it.
python -m shogi_ai.learn_loop \
    --initial-data data/initial_selfplay.jsonl \
    --rounds 6 --games 400 \
    --simulations 160 --search-mode gumbel --playout-cap-fraction 0.3 \
    --batch-size 32 --processes 10 --backend cshogi \
    --epochs 4 --lr 2e-3 \
    --round-lr 5e-4 --round-epochs 2 --latest-repeat 2 --grad-clip 1.0 \
    --eval-games 40 --eval-simulations 128 --gate 0.55 \
    --max-positions 600000
```

Expected wall time on the A100 instance was roughly 5-7 hours for
all six rounds (self-play ~2.5h, training ~1.5h, gating the rest); on
an L4, budget ~1.5x that, mostly in training and gating.  Start with
`--rounds 3` and extend by rerunning with a larger `--rounds` if the
session budget is tight.  Judge progress by the gate scores and by a
fixed side match against the depth-3 heuristic engine -- the first
run's champion scored 0.200 there, which is the number to beat first.

## Verification Commands

```bash
python3 -m unittest discover -s tests   # 148 tests; cshogi/torch tests skip when absent
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
