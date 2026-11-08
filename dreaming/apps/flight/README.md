# Flight simulator cockpit preview

Open FLIGHT on the desktop. W/S controls power, arrows pitch/bank, A/D rudder,
Enter restarts. Gentle runway contact lands; other ground contact crashes and
resets after a 30-step message. Physics, scenery and clouds are deterministic.

The 30x22-tile window contains a 224x152 first-person cockpit. The outside view
uses seeded terrain with sea, sand, green land, runway markings and occasional
clouds. Pitch moves the horizon; bank tilts it. Angled window struts, button
clusters and a dashboard frame three readable instruments: airspeed, attitude,
altitude. Gauges use 16 needle angles and 15 pitch/bank attitude-ball variants.
Charcoal dashboard, thick edged struts and dark dial faces frame the view.
Crashes crack the windscreen as well as displaying the reset message.
A minimal green strip shows altitude/speed/heading or the crash/reset message.

## Detailed tiles, zero model parameters

The outside scene is rasterized at pixel resolution and each 8x8 cell is matched
to the closest dedicated flight tile by pixel mismatch. The library contains
eight fills, 144 angled/offset two-colour edge tiles (horizon, coast, sand, cloud,
runway), two dither fills and eight rounded cloud parts. This keeps sub-tile boundary detail rather than
the rejected 4x4-pixel Paint-cell quantization. Cockpit art is fixed tile art;
needles and attitude indicators are small tile variants.

All art is deduplicated per 8x8 tile, so static gauge parts are shared across
needle angles and instruments; only crossed tiles vary. Flight uses 394 unique
tiles: 256 in app_flight and 138 at the start of app_spare (2304..2441).
Flight reserves at most the first 256 app_spare IDs (2304..2559), for a hard
512-tile total limit. No IDs present on main are renumbered.
Clouds use aligned rounded multi-lobed shapes. Distant runway details are
suppressed instead of becoming gray blocks; nearby runways retain white markings.

There are **zero new model parameters, heads, embeddings, observation keys or
sequence positions**. The 532 cockpit/view tokens are ordinary tiles within the
existing 768-token screen. model.py, dataset.py and train.py match main exactly.
Flight owns its reserved app state slot and extension fields 12..15.

Repository investigation: pictures.py draws an 8-colour 32x20 array, which Paint
maps through canvas_id into tiles with four 4x4 colour cells. No finer image-tile
helper exists on current main. The flight library instead reuses the atlas's
8x8 pixel-array convention and grid_to_pixels renderer, extending that same
fixed-art principle to detailed edges and gauges.

Driver patterns cover takeoff, cruise, turns, climbs, dives, landing attempts,
crashes and noisy interactions. Focused simulation: Driver(sim, seed, 'flight').
Preview endpoint: /sprites/flight. No training or Modal job is needed.
