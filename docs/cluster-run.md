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

Only phase `long` is supported. `--steps` overrides the default absolute
`--end-iter 10000`. The resolved start plus steps **must end on a multiple of
5000**, at or before `--iters`. Every checkpoint is at absolute iteration 10000,
15000, 20000, etc.; there are no pilot/gate chunks or extra calibration saves.
`train.py` also saves at the chunk end, so alignment is mandatory. Automatic
resume uses `/out/runs/<id>/ckpt.pt` and its run.json iteration; a fresh run uses
`/vol/runs/atlas_v2_first_5k/ckpt_005000.pt`. The source is never written.

## Schedule and calibration

Default batch32 × unroll6 × 24 ranks = 4608 frames/update. LR stays **4e-4,
unscaled**, with warmup restarted at iteration 5000 for 200 updates; cosine horizon
65000 and floor 0.1. Warmup multiplies the cosine. No TBPTT change is introduced.
`--lr`, `--warmup`, `--restart-iter`, `--min-lr-frac`, `--iters`, `--batch`,
`--unroll`, `--data-path`, `--extra-data`, `--extra-frac`, `--extra-shards`,
`--holdout-episodes`, and `--source` are exposed and locked by launch.json.
Default primary data is `/vol/shards`, cats extra `/vol/shards_cats_dense_v1`,
extra fraction .15, extra shard cap 40, holdout 2, sync-fix and sampler-v2 enabled.

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

Before launch check quota for all three nodes, source existence, data/holdout
settings, output identity, no concurrent writer, remaining budget and estimated
time to the next checkpoint. CPU tests and read-only previews do not demonstrate
actual distributed GPU execution; inspect RDMA/NCCL logs on the first real launch.
