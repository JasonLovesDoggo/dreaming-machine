# Opt-in picture-opening pilot

`modal run modal_picture_pilot.py` generates about 150k train and 15k held-out
action ticks, retaining whole episodes (so totals may slightly exceed targets).
It requires a clean committed checkout. Only the new `dreaming-picture-pilot-v1`
volume is mounted, writable at `/pilot`. There are at most three simultaneous
CPU generation functions, each with 16 cores and 16 worker processes; no GPUs.
Existing driver modes, training, and active corpora are untouched.

For a local sample:

```sh
python -m dreaming.picture_pilot --out pilot-local --split train --steps 1000 --workers 4
```

Each episode begins at the desktop, navigates with existing driver primitives,
opens exactly one existing/new image, records stages 1–10, holds it complete for
10–20 ticks, then optionally focuses another window or edits Paint. Cursor
movement and idle controls are included. Half of the filename schedule covers
DOG, PEACH, ORANGECAT and ORANGE CAT; the other half cycles current package
categories with varied keywords, suffixes and extensions. The simulator's hidden
episode-seeded picture RNG is unchanged; filenames do not determine pixels.

Reserved seed namespaces are `[200000000, 210000000)` for train and
`[210000000, 220000000)` for held-out, separate from the existing 100000000
evaluation namespace and the documented primary/special/cat generation bases.
The two train jobs use offsets 0 and 100000; held-out starts at its own base.

Shards have exactly dataset.py's observation/action keys plus `ep_len`, with
one zero-padded action per episode. `ep_len` includes that padded observation;
reported ticks exclude it. Sidecars contain source SHA, hashes of os_sim.py,
driver.py and every pictures Python module, seeds, opened filenames/categories,
per-episode action hashes, load boundaries, completed-canvas hashes and counts.
Every generated episode is replayed from its seed and recorded actions, checking
every observation field and the completed pixels against the cached image
produced by the current generator. Validation fails closed.

`loading_ticks` counts states 1–9; `reveal_updates` includes completion stage 10.
`newly_revealed_cells` counts grid tiles (160 per open), not logical pixels.
`changed_cells` and `idle_cells` partition all 768 grid cells per action tick;
`idle_ticks` counts grid-unchanged frames, not necessarily zero-input actions.
Paint-edit ticks have a paint_color or paint_stroke event. Summary JSONs include
per-name and per-category open counts. Per-partition and combined contact sheets
show completed, unedited canvases labeled by filename and episode seed.

Outputs are under `train/part_00`, `train/part_01`, `heldout/part_00`; the root
`manifest.json` aggregates counts and verifies split seed disjointness. The root
`contact_sheet.png` combines the three partition sheets. Generation refuses to
overwrite existing shards; use a separate new destination for another pilot.
