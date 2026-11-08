# Dreaming Machine

A deterministic tile-based micro OS (desktop, file explorer + trash, Paint, Solitaire, Doom-lite) driven by a
scripted "smart user", used to train a world model that predicts the next screen from the current screen + input.

- Screen: 32x24 tiles of 8x8 px (256x192). Every tile comes from a fixed atlas (`dreaming/tiles.py`), so a frame is
  exactly 768 integer tokens. Text uses a 3x5 font with two characters per tile.
- Doom-lite: 160x120 textured raycaster, encoded as 160 column tokens (wall type, side, height, texture u) + 4 sprite
  tokens + a meta token (gun flash, viewport position).
- Cursor: separate x/y tokens. Action per step: (mouse x, mouse y, left button, key).
- Model (`dreaming/model.py`): transformer over tiles + cursor + action + doom tokens + recurrent memory tokens +
  OS-state query tokens (auxiliary loss). Predicts all next-frame tokens in one pass.

```bash
python -m dreaming.dataset --out data/local --steps 400000          # ~40s on 8 cores, ~16 MB
python -m dreaming.train --data data/local --out runs/local         # GPU recommended
python -m dreaming.demo --ckpt runs/local/ckpt.pt --data data/local  # web UI (dreaming/web), real vs neural

# Modal
modal run modal_app.py::gen_data --total-steps 20000000
modal run modal_app.py::train --gpus 8 --iters 30000
modal run modal_app.py::fetch
```

### Demo telemetry

The demo sends anonymous page views, visible-page duration heartbeats (15 seconds),
control/action counts, active-app frame counts, checkpoint names, client FPS and
simulator/model step latency to `POST /telemetry/events`. It never records typed
text or raw IP addresses; addresses are salted SHA-256 hashes truncated to 16 hex
characters. Visitor IDs persist in localStorage; each page load gets a session ID.
User agents are retained. `GET /telemetry/summary` is a public, aggregate JSON view
(durations in seconds, latency in milliseconds; checkpoints count reports).

JSONL files live in `./runs/telemetry` locally and `/vol/telemetry` on Modal, with
the existing data volume committed every 30 seconds while there are new events.
Set `DREAMING_TELEMETRY_DIR` to override the directory and optionally set
`DREAMING_TELEMETRY_SALT` for a stable private salt; otherwise a salt is generated
in that directory. Last uncommitted events may be lost if a container is killed.
Summaries read files visible to the current container (no cross-container reload).
There is no retention/rotation beyond daily files; remove old JSONL files as needed.

### Training metrics

`/training` (footer link "training metrics") plots `<runs>/<run>/metrics.jsonl` for every
run directory directly under the runs root (`/vol/runs` on Modal, `DREAMING_RUNS_DIR` or
`./runs` locally), one line per run. Each line is a JSON object:

* `{"kind": "train", "it": 25, "loss": ..., "grid": ..., "cursor": ..., "doom": ..., "state": ...,
  "acc_all": ..., "acc_changed": ..., "frac_changed": ..., "lr": ..., "grad_norm": ..., "ss_prob": ...,
  "samples_per_sec": ...}`: every numeric field is plotted; string fields and flat string
  dicts (e.g. `source`, `aggregation`) are shown as run notes.
* `{"kind": "eval", "it": ..., "metrics": {app: dreaming.eval result}}`: held-out per-app
  changed-tile accuracy, grid CE, all-tile accuracy vs copy-last baseline, and rollout
  accuracy vs copy-first baseline (mean and final step per horizon, plus the latest per-step curve).

`GET /training/runs` lists runs and `GET /training/metrics?run=<id>` returns one run's thinned
columns. Only `metrics.jsonl` files are opened (no symlinks, no checkpoints); malformed lines and
non-finite values are skipped and counted; files are parsed incrementally, reading at most the
last 16 MB. On Modal the dashboard reloads the volume in a background thread at most once a
minute, only while someone has the page open; reload failures are shown on the page.
The chart library is uPlot 1.6.32 (MIT), vendored in `dreaming/web/static/vendor/uplot`.
