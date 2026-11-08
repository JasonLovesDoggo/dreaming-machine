# Isolated 24-B300 training

The `dreaming-cluster` app requests 3 RDMA nodes, each with 8 B300s, 64 CPU
cores and 262144 MiB RAM. Importing the module does not launch/hydrate anything.
Training and eval mount `dreaming-computer-data` read-only at `/vol`; all their
writes go to a distinct `dreaming-run-<id>` volume at `/out`.

## Operator commands

Authenticate Modal 1.6.3 and run from the repository root. Choose a new lowercase
run id, 3–48 characters, never a protected/existing training run. Only one launch
or eval may write a given output volume at a time.

```sh
export DREAMING_RUN_ID=atlas-cluster-v1
# Read-only SDK preview: no app initialization, volume creation, or functions:
python -c 'import cluster_run; cluster_run.launch(dry_run=True)'
# Allocates 24 GPUs; default resumes at 5000 and targets absolute iteration 10000:
DREAMING_RUN_ID=atlas-cluster-v1 modal run --detach cluster_run.py::launch
# Alternative tighter budget for the initial 5000-update chunk:
# DREAMING_RUN_ID=atlas-cluster-v1 modal run --detach cluster_run.py::launch --end-iter 10000 --budget-hours 1
# After the trainer has exited, evaluate (CPU by default):
DREAMING_RUN_ID=atlas-cluster-v1 modal run cluster_run.py::eval --ckpt-name ckpt_010000.pt
# Optional GPU evaluation: append --gpu (DREAMING_EVAL_GPU selects type, default B300).
# Resume later, keeping the same schedule/data lock:
DREAMING_RUN_ID=atlas-cluster-v1 modal run --detach cluster_run.py::launch --end-iter 45000 --budget-hours 6
# CPU-only explicit mirror/publication:
DREAMING_RUN_ID=atlas-cluster-v1 modal run cluster_run.py::mirror
DREAMING_RUN_ID=atlas-cluster-v1 modal run cluster_run.py::publish --ckpts ckpt_010000.pt,ckpt_015000.pt
```

Do not execute all alternatives blindly: initial default reserves the full 8-hour
cap. Split that reservation explicitly if planning multiple launches. Preview
prints the complete config and torchrun command before remote launch. Avoid
`modal run --dry-run`: Modal app initialization may hydrate/create resources.
The direct Python preview queries volume metadata/files through the SDK only.

Checkpoint cadence is configurable with `--ckpt-every` (positive int, default 5000); the end must align to it, and cadence may change on resume.
Only phase `long` is supported. `--steps` overrides the default absolute
`--end-iter 10000`. The resolved start plus steps **must end on a multiple of
the cadence (5000 by default)**, at or before `--iters`. By default, every checkpoint is at absolute iteration 10000,
15000, 20000, etc.; there are no pilot/gate chunks or extra calibration saves.
`train.py` also saves at the chunk end, so alignment is mandatory. Automatic
resume uses the numbered `/out/runs/<id>/ckpt_NNNNNN.pt` matching run.json,
never the potentially partial rolling ckpt.pt. A missing numbered file fails
instead of falling back. A fresh run uses
`/vol/runs/atlas_v2_first_5k/ckpt_005000.pt`. The source is never written.

## Schedule and calibration

Default batch32 × unroll6 × 24 ranks = 4608 frames/update. LR stays **4e-4,
unscaled**, with warmup restarted at iteration 5000 for 200 updates; cosine horizon
65000 and floor 0.1. Warmup multiplies the cosine. No TBPTT change is introduced.
`--lr`, `--warmup`, `--restart-iter`, `--min-lr-frac`, `--iters`, `--batch`,
`--unroll`, `--data-path`, `--extra-data`, `--extra-frac`, `--extra-shards`,
`--holdout-episodes`, and `--source` are exposed and locked by launch.json.
Default primary data is `/vol/shards`, cats extra `/vol/shards_cats_dense_v1`,
extra fraction .15, extra shard cap 40, holdout 2, sync-fix and compile enabled.
Sampler-v2 is disabled by default; opt in with `--sampler-v2`. This choice is
included in launch.json and cannot change on continuation launches.

Rank 0 inspects metrics at each 300-second tick. Once at least 300 updates have
been logged, it prints one calibration line with samples/s, updates/s, loss,
grad_norm, projected end timestamp and kill_at. Any new train row with null or
non-finite loss/grad_norm, or post-warmup loss above `--guard-max-loss` (2.0),
terminates the torchrun process group without a new checkpoint or relaunch.
Projection past kill_at warns only. Calibration happens in-process; it is not a
separate short training chunk. Inspect its output and throughput promptly.

## Memory, budgets, publication and failures

Primary data: 58M frames × ~2275 B ≈132 GB /24 ranks ≈5.5 GB/rank, ~44 GB/node
resident. Allow ~2× transient memory during loading plus CUDA contexts, extra cats
shards, model and optimizer. Partitioning is mode-balanced. OMP defaults to 2
threads. Rendezvous uses node 0 IPv4, port 29500; NCCL_SOCKET_IFNAME defaults eth0.

Each launch is capped at 8 hours, with a default 8-hour total allocation ledger
cap (`--total-cap-hours` explicitly overrides the total). Missing end records
charge their entire reserved deadline. These are wall-clock allocation hours;
one hour equals 24 GPU-hours. Eval is separate and not charged to that ledger.
Budget deadlines include scheduling delay. Training stops 600 seconds early to
allow cleanup. SIGTERM/interrupts/errors stop the process group, escalating to
SIGKILL after 60 seconds. No forced checkpoint occurs at timeout: work since the
last 5000-update save may be lost. No Modal retries or elastic restarts are enabled.

Rank 0 commits every 300 seconds, spawns the CPU mirror, and publishes newly
available numbered checkpoints at the 5000 cadence. Final cleanup records ledger
end, commits and waits for a final mirror. A hard container loss can lose writes
since the last commit. Inspect every node return code and logs before resuming;
never erase ledger reservations or edit launch.json to bypass mismatches.

Only mirror_metrics mounts the main volume writable. It atomically merges train
then eval metrics, copies selected numbered checkpoints atomically, and writes
`run.json` **last**, with atlas_version=2 and the maximum published iteration.
This makes published checkpoints discoverable in the web selector. It refuses
foreign files; permitted files are metrics.jsonl, run.json, ckpt_NNNNNN.pt and
their .tmp files. It never publishes ckpt.pt or writes protected run directories.
Copies are gated by the output run.json iteration, which is written only after
both checkpoint saves finish. Published checkpoint files are immutable: an existing
same-size file is skipped, and a size mismatch fails instead of overwriting it.
The web metrics parser accepts nested eval rows following train rows.
Eval writes separate `eval/metrics.jsonl`; run it after training exits to avoid
concurrent output-volume writers. Eval defaults to CPU; `--gpu` explicitly opts
into one GPU, with `DREAMING_EVAL_GPU` changing its B300 type default.
Eval requires an explicit numbered checkpoint at or before the output run.json
iteration. The CPU mirror has 2048 MiB RAM and an 1800-second timeout for large copies.

Before launch check quota for all three nodes, source existence, data/holdout
settings, output identity, no concurrent writer, remaining budget and estimated
time to the next checkpoint. CPU tests and read-only previews do not demonstrate
actual distributed GPU execution; inspect RDMA/NCCL logs on the first real launch.

## Picture-aware fork (operator launch only)

First stop the old training app and wait for **every** source ledger allocation
to have an `end`. The launcher refuses active or missing/empty source ledgers;
do not fill in end timestamps manually. Choose a fresh target run id/volume.
The source numbered checkpoint must exist and be at or before its run.json iter.
The old output volume is mounted read-only at `/src`, and the independent picture
volume is mounted read-only at `/pic`. Neither is written by the fork.

```sh
export DREAMING_RUN_ID=overnight_20261108_pic24k
export DREAMING_SOURCE_RUN_ID=overnight_20261108_b30024
export DREAMING_PICTURE_VOLUME=dreaming-picture-pilot-v1
# Read-only SDK preview (no app launch/volume creation):
python -c 'import cluster_run; cluster_run.launch(source_iter=24000, extra_shards=400, lr=0.0004, warmup=200, min_lr_frac=0.1, restart_iter=5000, iters=65000, batch=32, unroll=6, end_iter=65000, ckpt_every=1000, budget_hours=1, stop_at="2026-11-08T14:29:00Z", picture_data="/pic/train/part_00,/pic/train/part_01", picture_frac=.15, dry_run=True)'
# This command allocates GPUs; execute only after reviewing the preview:
modal run --detach cluster_run.py::launch --source-iter 24000 --extra-shards 400 --lr 0.0004 --warmup 200 --min-lr-frac 0.1 --restart-iter 5000 --iters 65000 --batch 32 --unroll 6 --ckpt-every 1000 --end-iter 65000 --budget-hours 1 --stop-at 2026-11-08T14:29:00Z --picture-data /pic/train/part_00,/pic/train/part_01 --picture-frac 0.15
```

Set an appropriate future UTC stop time and allocation budget: the source ledger
is copied with `inherited_from` into the new ledger and counts against the same
8-hour total cap. The deadline is the earlier of stop time and requested budget;
less than 30 minutes remaining is refused (the 600-second cleanup reserve remains).
This ledger check is not an atomic cross-app lock: operators must prevent racing
launches/restarts of the source or multiple forks of it.

The source `launch.json` is required. Every locked setting must match it except
the checkpoint source path and picture data/fraction. Mismatches name the keys
to correct; in particular, preserve `--extra-shards 400` rather than its default 40.

This resumes iteration and optimizer state at 24000 without restarting the LR
schedule: the anchor remains 5000, warmup 200, peak LR 4e-4, horizon 65000, floor .1.
The dashboard carries the original 960000-frame base at iteration 5000.
Pictures use **all** shards from the explicitly listed `/train/` directories on
every rank; no partition or last-shard drop. Paths resolving outside `/train/` or
containing `heldout` are refused. `--picture-holdout-episodes` defaults to 0 because
the volume has its own independent heldout split. Never supply `/heldout` paths.
With batch32 and cats/pictures .15/.15, batches contain 22 primary, 5 cats and 5
pictures; cats retain exactly their existing share. Pictures require `--extra-data`.
Without pictures the loader and torchrun command are unchanged. Picture directory
and fraction settings are locked; legacy non-picture locks remain compatible.

For a later continuation of the **target**, omit `--source-iter` (forks require a
fresh target), keep the three environment variables and picture settings, and pass
`--source /src/runs/overnight_20261108_b30024/ckpt_024000.pt` to match the target's
source lock. Resume then uses the target's own completed numbered checkpoint and
already inherited ledger rather than copying the source history a second time.

## Automatic continuation (no quality pause)

`DREAMING_RUN_ID=<id> modal run --detach cluster_run.py::auto_continue --after-iter 20000 --end-iter 65000 --ckpt-every 1000 --stop-at 2026-11-08T14:30:00Z`
starts a 1-CPU controller. It records the ledger, then checks every 30 s for up to 90 min until the active ledger entry has ended and both the output and main `run.json` show `--after-iter`. After a 30 s handoff it calls `train_cluster.remote` exactly once. The config comes from the saved `launch.json` (LR, horizon, extra shards and the rest), the run resumes from `ckpt_<after>.pt`, and the budget is `min(8h total cap - sum of ledger durations, stop_at - now) - 1s`. It refuses to run if the ledger changed, `continuation.json` already exists, or the run is past the boundary. Before launching it appends a dashboard `config` row (`phase_end`, `checkpoint_every`) to `metrics.jsonl`.
