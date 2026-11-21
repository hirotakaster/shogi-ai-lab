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
- `shogi_ai/network.py`: the single definition of the policy/value net (ResNet presets small/base/large, plus the experimental `attn` hybrid that interleaves self-attention blocks into the base trunk), plus self-describing checkpoints. Checkpoint configs carry `attn_every`/`attn_heads` with defaults of 0/8, so pre-hybrid checkpoints keep loading as pure ResNets and `CHECKPOINT_VERSION` did not need a bump (the encoding contract is unchanged).
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
- `colab/train_github.ipynb`: the training notebook (clones the newest
  `mochi/batch-*` branch from GitHub; Drive holds artefacts only).

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
- Self-play can ban named moves at the root on named plies
  (`--banned-moves 0:2h5h,1:8b5b` in selfplay_az and learn_loop).
  `MCTS.run(..., banned_indices=...)` drops those policy indices
  from the root's edge lists before the search, so a banned move is
  never searched, played or recorded; the restriction is applied on
  every run, which is what makes it work for a retained subtree
  adopted as the next root (it was expanded as an ordinary leaf
  with every move present).  Only the root is restricted -- deeper
  nodes are other positions, where the same policy index is a
  different move.  A ban covering every legal move is ignored.
  Gating and real play never pass a ban.
- `MCTS.run(..., simulations=N)` overrides the budget for one call only;
  playout-cap randomisation is built on it.
- PUCT selection supports first-play urgency (`--fpu-reduction` in
  self-play, evaluate and learn_loop; unset is the legacy behaviour).
  An unvisited edge otherwise scores q = 0.0 -- a draw -- whatever the
  position is worth, so in a position worth -0.6 every untried move
  looks better than the position deserves.  With a reduction set, an
  unvisited edge instead inherits the node's own mean value minus the
  reduction (the KataGo/Leela estimate).  It applies below the root in
  both modes and at the root in `puct` mode; the Gumbel root already
  completes unvisited actions with the root value.  Edges carrying a
  virtual loss are exempt -- they are being visited right now and must
  keep looking like losses.  Opt-in: A/B it at the gate before adopting.
- `--round-max-steps N` in learn_loop caps the optimiser steps of a
  round that trains from a champion (train_az `--max-steps`).  With
  full epochs the step count grew with the replay buffer -- ~500 to
  ~2,000 steps per round over the 2026-11-14 run -- and the gate
  scores swung with it (0.350/0.412 rejections among steady
  promotions).  A fixed budget keeps every round's update the same
  size; train_az sizes the OneCycle schedule to the cap so the cycle
  still completes inside the budget.
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

## Campaign 2 (settings validated 2026-11-15)

Campaign 1 (2026-11-14/15, L4, Batch 8 code) completed all 40 rounds:
22 promotions, final champion 41,892 steps.  The yardstick match
against the depth-3 heuristic improved to **0.338** (13W-26L-1D,
-117 Elo) from the first champion's 0.200, and the final champion
beat campaign 1's round-1 candidate 0.850 (+301 Elo) -- real progress,
still below the toy heuristic engine.  The next yardstick target is
0.500: actually beating the heuristic.

Before campaign 2, the open search settings were A/B-tested on a T4
(100 games per arm, final champion vs the round-1 candidate, one
seed): baseline (200 simulations, `--gumbel-candidates 16`) 0.850;
`--fpu-reduction 0.2` scored 0.750 -- harmful, do not use;
**`--gumbel-candidates 8` scored 0.895 and was the fastest arm**;
400 simulations scored 0.765 at 16 candidates and 0.845 at 8 --
extra simulations bought nothing at either candidate count, so
simulations stay at 200.  The bottleneck is the network's evaluation
quality and the volume/freshness of self-play data, not search depth.

Campaign 2 therefore continues from the campaign-1 champion on the
base network with a fresh replay buffer (restore only
`policy_value.pt` from the Drive backup; the notebook has a restore
cell for this and cells 19/20 are skipped), with these changes from
campaign 1: `--gumbel-candidates 8`, `--eval-games 80` (halve the
gate's sampling noise), and `--round-max-steps 1200` (campaign 1's
training steps per round drifted ~500 -> ~3,000 as the buffer grew;
the cap keeps every round's training uniform).  The notebook's learn
cell already carries exactly this configuration.

Campaign 2's first 10 rounds ran on 2026-11-15 (L4, 3 h 7 min):
2 promotions, champion 41,892 -> 43,052 steps (one promotion in
rounds 1-7 -> 42,242; round 8 promoted at 0.575, +53 Elo).  Round 9
was rejected at 0.400 and round 10 at 0.494 (-4 Elo) -- the
candidates had become champion-level but not clearly better.
Per-round detail for rounds 1-7 scrolled off the Colab output; the
full record is in the Drive backup's history.json.  Two process
lessons: the loop's backup overwrote campaign 1's artefacts in
place (the campaign-1 final champion was never archived under a
milestone name, so it can no longer be matched head-to-head), and
the restore path silently depended on `PRESET` being defined by the
speed or pretrain cell.  Batch 11 fixes both: the notebook archives
the Drive champion to `ARTIFACTS/champions/<name>.pt` at campaign
milestones, the restore cell defines `PRESET` itself, and a
yardstick cell measures the restored champion against the depth-3
heuristic BEFORE any further training.  New campaigns should also
get their own backup subfolder instead of sharing
`shogi-ai-lab-artifacts`.  Batch 11 also raises the per-round
training amount, which the round timing showed was nearly free
(self-play ~900 s and gating ~180 s per round vs ~35 s of
training): `--round-epochs 5`, `--latest-repeat 4` (the newest
round is only ~12k of a ~132k-position buffer), and
`--round-max-steps 3600`.  Batch 12 fixes the yardstick cell itself:
on 2026-11-15 the single-process yardstick, piped through `tail -3`
so nothing printed until it finished, ran 67+ minutes with no visible
progress; the cell now runs `--processes 10` and streams per-game
progress.  Match conditions are unchanged (same games, seeds and
tally -- `evaluate --processes` is parallel-invariant by the seeding
rule above), so the result stays comparable with the 0.338 baseline.
Batch 13 adds the experimental `attn` preset (the user asked for a
Transformer direction on 2026-11-15): the base trunk with a pre-norm
self-attention block (8 heads, learned positional embeddings over the
81 squares) after blocks 5 and 10, ~8.09M params vs base's 7.47M.
It is opt-in only -- the notebook and the running campaign stay on
`base` -- and every CLI accepts it through `--preset attn` or by
loading an `attn` checkpoint (checkpoints are self-describing).
Switching a campaign to it forks the lineage like the large switch
would: no weight transfer, the bridge is distillation pretraining
from self-play records, which are architecture-independent.  That
distillation experiment ran on 2026-11-15; see Campaign 3 below.

Campaign 2's extension (rounds 11-20, 2026-11-15, L4, Batch 11
settings, 3 h 31 min) promoted only round 18 (0.575), ending at a
46,652-step champion with a 279,762-position buffer; the Batch 11
training increase did not raise the promotion pace (1/10 in the
extension vs 2/10 in rounds 1-10).  The yardstick then delivered the
campaign's most important lesson: the 46,652-step champion scored
**0.425** (17W-23L-0D, -53 Elo) against the depth-3 heuristic, DOWN
from the 43,052-step champion's 0.700, even though it had beaten
that same predecessor head-to-head at the round-18 gate.  Internal
gating alone can therefore drift away from external strength -- the
candidates increasingly adapt to the incumbent's style (a
"champion-copy" loop).  From campaign 3 on, measure the yardstick
every ~10 rounds, not only at the end.

## Campaign 3 (attn, decided 2026-11-16)

The phase-2 architecture experiment ran on 2026-11-15 instead of
waiting: a fresh `attn` network was distilled from campaign 2's
self-play records (243,264 loadable positions of 279,762 recorded;
20 epochs, 18,060 steps, ~16 min on L4; final val policy 2.0607 /
value 0.5824 / top1 0.488), archived at
`ARTIFACTS/attn/policy_value_attn_distilled.pt`.  Measured the same
night: yardstick **0.450** (18W-22L-0D, -35 Elo) and head-to-head vs
the base champion **0.494** (37W-38L-5D, -4 Elo, 80 games at gate
settings) -- statistical parity.  The architecture fork is therefore
free: full strength was recovered from data alone, with no weight
transfer.  No strength gain yet either; attn's ceiling is testable
only by its own self-play.

Decision (2026-11-16): campaign 3 runs on attn.  It starts from the
distilled model as its initial champion with a fresh buffer, backs
up to its own subfolder `ARTIFACTS/campaign3/`, and keeps campaign
2's training settings (Batch 11 amounts, Gumbel candidates 8, gate
80 games); the first block is 10 rounds, with a yardstick
measurement every 10 rounds.  The notebook carries campaign-3
restore/learn/check cells (Batch 14); do not use the campaign-2
restore or save cells for it.  The base champion and the
`campaign2_r10_43052` milestone stay archived as the fallback.

Campaign 3 block 1 (rounds 1-10, 2026-11-17/18, L4) promoted
**nobody**: the gate scores were 0.438 / 0.406 / 0.500 / 0.469 /
0.438 / 0.506 / 0.463 / 0.537 / 0.463 / 0.469, and the distilled
model stayed champion throughout (the round-10 yardstick was
therefore pointless and was skipped -- that champion is the file
already measured at 0.450).  Candidates trained longer every round
(240 -> 2,990 steps) on a buffer growing to ~123k positions, yet
the scores never trended up: a round's candidate is a light
finetune of the champion on the champion's own self-play, so it
converges to a champion copy, and the 0.55 gate then discards the
small genuine gains along with the noise.  Block 2 (rounds 11-20)
therefore changes the recipe, not the architecture (more attention
blocks were considered and deferred -- at this data volume the
base/attn distillation parity says architecture is not the binding
constraint): `--gate 0.53` with `--eval-games 200` (a looser
threshold measured more tightly, sigma ~0.035 instead of ~0.056,
at ~+6 min of gating per round), and a real per-generation
training pass instead of a finetune -- `--round-lr 1e-3`,
`--round-epochs 10`, `--round-max-steps 6000`.  Promotion still
requires beating the champion at the gate (always-promote was
considered and rejected by the user); the block ends with a
yardstick measurement of the final champion, and a bad regression
rolls back to the distilled model.  The notebook carries block-2
restore/learn cells (Batch 15); in a session continuing from
block 1 only the learn cell is needed, because the loop skips the
rounds recorded as finished in history.json.  Also queued for the
block-2 wrap-up: measure how concentrated the buffer's openings
are (per-ply SFEN duplication) before deciding whether self-play
needs a hotter opening temperature for diversity.

Campaign 3 block 2 (rounds 11-20, 2026-11-18, L4) promoted
**once**: round 15 scored 0.532 and the champion changed to
24,060 steps -- the campaign's first promotion, and the
self-play generator unfroze with it.  The other scores were
0.448 / 0.480 / 0.497 / 0.460 / 0.485 / 0.527 / 0.517 / 0.520 /
0.497 (block mean 0.496 vs block 1's 0.470; rounds 17-19
clustered just under the gate).  The final champion's yardstick
was **0.400** (16W-24L, -70 Elo) against the depth-3 heuristic
-- below the distilled model's 0.450 but inside 40-game noise
(sigma ~0.08), and the head-to-head gate had said +23 Elo, so
an external gain is unconfirmed: the campaign-2 gate/yardstick
divergence at a smaller amplitude.  Round 15's training log
also settled the epochs question: the 6,000-step cap stopped
it at epoch 8/10 with val top1 improving 0.476 -> 0.542, so the
heavier regime is not overfitting; the cap, not the epochs, is
the binding limit (~7,000 planned steps at this buffer size).

The queued opening measurement came back extreme: across the
20 self-play files (~2,000 recorded positions per ply), ply 2
has only **2 distinct positions** (top line 99.9%), the ply
4/6/8 top-1 shares are 74-93%, and real branching starts only
around ply 10.  Temperature 1.0 does not explore when the
policy is this peaked; the campaign's 8,000 games funnel
through essentially one opening line.  Block 3 (rounds 21-30)
therefore keeps the 24,060-step champion (decided 2026-11-18:
rollback was the pre-agreed branch for a yardstick below
0.450, but the head-to-head and the diversity data say the
champion is not the problem -- the data distribution is, and
rolling back would not change it) and attacks the funnel:
two-stage temperature (Batch 16: `--opening-moves 10
--opening-temperature 1.5`, then the regular temperature
window) and `--round-max-steps 10000`, so the epochs bind
instead of the cap.  The block-3 restore cell first archives
the champion as `champions/campaign3_r15_24060.pt`.  After
block 3: measure the yardstick again, and re-measure the
opening diversity to check the funnel actually widened.

Campaign 3 block 3 (rounds 21-30, 2026-11-18/19, L4) promoted
**once**: round 24 scored 0.535 and the champion changed to
34,060 steps.  The other scores were 0.500 / 0.522 / 0.487 /
0.440 / 0.510 / 0.500 / 0.505 / 0.472 / 0.492 (block mean
0.496, identical to block 2's).  As run, the opening
temperature was **2.0**, raised from the notebook's 1.5 at
launch by the user (temperature on visit counts flattens
weakly: 1.5 would still have left ~75% of games on the same
first move).  Every candidate trained the full 10,000-step
cap (val top1 0.57-0.60).  The final champion's yardstick
jumped to **0.550** (22W-18L-0D, +35 Elo) -- the campaign's
first above-0.500 external score, up from the previous
champion's 0.400.  The diversity fix verified on the rounds
21-30 files (124,793 positions): ply-2 distinct positions
2 -> 13 (top-1 still 95.6%: the first-move pair is the last
funnel point), ply 4 top-1 78.9% -> 44.9%, ply 10 distinct
68 -> 324, near-total diversity by ply 15.  Internal gate
mean flat while the external yardstick jumped: diversity
bought generalisation the internal gate cannot see.  All
state is in `ARTIFACTS/campaign3`; the 24,060-step champion
is archived as `champions/campaign3_r15_24060.pt`.

Block 4 (rounds 31-40) presses on data volume and freshness,
the constraint every experiment so far points at (search
scaling failed the T4 A/B, distillation reached parity from
data alone, block 3's gain came from a data-distribution
change): 800 self-play games per round; a temperature
curve (Batch 17: `--temperature-points
0:4.0,4:4.0,39:1.0,50:1.0,51:0.0` -- flat at 4.0 for the
first four plies, a geometric decay to 1.0 by ply 39, held
to ply 50, greedy after; the net is nowhere near
discovering joseki, its crystallised main line is early
noise, so breadth comes first, into the early middlegame);
the buffer as a 500,000-position window
(`--max-positions` -- train_az's cap keeps the newest
positions, so the window slides over generations).  The
buffer starts trimmed: rounds 1-20 self-play
(pre-diversity-fix, funnelled through a single opening
line) is excluded -- the block-4 restore cell deletes
those files locally after restoring; they remain in the
Drive backup (the backup is copy-only), so the trim is
reversible.  The buffer therefore starts from rounds
21-30 (~125k positions).  And the
gate's 200 games recorded as training data
(`--record-gate-games`; `gate_NNN.jsonl` joins the buffer
from the next round).  Everything else carries over from
block 3.  The block-4 restore cell first archives the
champion as `champions/campaign3_r24_34060.pt`.  ~70
min/round, ~12 h for the block; the loop resumes across
sessions.

Block 4 as run extended to round 45 in three sessions
(2026-11-20/21).  Rounds 31-35 on the original curve promoted
R32 (0.642), R34 (0.535) and R35 (0.560) -- champion 64,060
steps, block gate mean 0.540 -- yet that champion's yardstick
regressed to **0.312** (12W-27L-1D, -137 Elo), the strongest
gate/yardstick divergence yet.  The diversity re-measure
cleared the curve's head: rounds 31-35 openings hit the target
(ply-2 top-1 58.3%, 64 distinct), so the suspicion falls on
the curve's depth -- still 1.43 at ply 30 and 1.0 held to
ply 50, semi-random middlegame play corrupting the targets.
Rounds 36-40 therefore ran a cooler curve
(`0:4.0,4:4.0,12:1.0,30:1.0,31:0.0`: the hot head kept, the
body back to block 3's shape) from the 64,060 champion, with
rounds 31-35 self-play excluded from the buffer (archived
locally; Drive copies intact) and their gate games kept.
R37 (0.532) and R40 (0.535) promoted -- champion 84,060
steps -- and the yardstick recovered to **0.500** (20W-20L),
meeting the pre-agreed ~0.45-0.50 bar, so no rollback.
Rounds 41-45 promoted R42 (0.530) and R45 (0.545) -- final
champion **104,060 steps**, campaign total 9/45 -- but the
yardstick fell again to **0.362** (14W-25L-1D, -98 Elo).
The user's call: do not fret over the gate; no rollback was
made (the 84,060-step champion stays archived as
`champions/campaign3_r40_84060.pt`).

Counting the recorded first moves (rounds 21-30 + 36-45
files) then found the funnel's last point: sente's first
move is the central rook swing 2h5h in **82.2%** of recorded
games, the pair 2h5h + 8b5b in **65.5%**, with only 13
distinct ply-1 positions -- the temperature curve widened
everything except the very first move.  Block 5 (rounds
46-50, Batch 19) therefore bans it in self-play
(`--banned-moves 0:2h5h,1:8b5b`), keeps the cooler curve
and the same buffer composition, and restores selectively
(the block-5 restore cell copies only self-play rounds
21-30 + 36-45 and gate rounds 31-45, and archives the
104,060-step champion as `champions/campaign3_r45_104060.pt`).

## Verification Commands

```bash
python3 -m unittest discover -s tests   # 202 tests; cshogi/torch tests skip when absent
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
