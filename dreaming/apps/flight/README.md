# Flight simulator (simplified, learnable)

Open FLIGHT on the desktop. W/S throttle, Up/Down pitch, Left/Right bank, A/D rudder,
Enter resets. Gentle contact with a runway lands; water, sand or grass crashes and the
windscreen cracks until reset.

## World and view

The world is deliberately tiny so each frame depends on a handful of numbers
(attitude, heading, altitude, position relative to the islands):

- flat ocean and sky,
- two flat elliptical islands (`AIRFIELDS`) with sand rim, grass and a marked runway,
- one to three clouds placed by a seeded RNG.

`render.py` projects this scene at full pixel resolution into a 224x152 first-person
cockpit (dark frame, angled struts, 24-px dashboard with speed / attitude / altitude
gauges) and snaps each 8x8 cell to the closest dedicated flight tile (fills, two-colour
edge tiles at quantized angles/offsets, cockpit and gauge tiles). Everything is
deterministic per seed; resets pick (reproducibly, per seed and episode) runway or
approach spawns at either airfield.

## Zero model parameters

Flight is drawn entirely with ordinary tile IDs inside the existing 32x24 grid: no new
heads, embeddings, observation keys or sequence positions, and `model.py`, `dataset.py`
and `train.py` are unchanged. Its 719 unique tiles live in `app_flight` (256) plus the
first IDs of `app_spare` (Flight claims at most 2304..3071); no existing ID moves and the
palette is unchanged. State uses the reserved app slot plus ext fields 12..15
(altitude, speed, throttle, crash).

## Driver

`Driver.task_flight` runs a short randomized closed-loop autopilot session
(`FLIGHT_STEPS`): takeoffs, cruise, left/right turns, climbs/dives, landing approaches
with go-arounds, deliberate crashes, resets and random key presses. Its mixed-mode weight
is kept low so Flight stays a small share of mixed frames. Focused data:
`Driver(sim, seed, 'flight')`. Preview sheet: `/sprites/flight`.
