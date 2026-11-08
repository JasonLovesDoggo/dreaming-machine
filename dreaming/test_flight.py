import numpy as np
import torch
from .apps import flight as fl
from .apps.flight.render import render_scene, match_tile, ART
from .os_sim import MicroOS, obs_to_pixels, STATE_VOCAB, KEY_PLUS, DESK_ICONS
from .tiles import VOCAB, ATLAS, grid_to_pixels
from .model import WorldModel, Config, compute_loss, decode
from .dataset import OBS_KEYS
from .flight_art import SKY, SEA, GRASS, FRAME_COLORS, FILLS, PAIRS


def _runway_start(seed=3):
    f = fl.Flight(seed)
    while f.airborne:
        f.reset()
    return f


def test_flight_takeoff_crash_reset_and_determinism():
    a, b = _runway_start(), _runway_start()
    events = []
    for key in ['W'] * 30 + [''] * 25 + ['UP'] * 14 + [''] * 60:
        out = a.step(key)
        assert out == b.step(key)
        events += out
    assert a.alt > 20 and 'flight_takeoff' in events
    assert np.array_equal(fl.render_tiles(a), fl.render_tiles(b))
    a.x, a.y, a.alt = 400., 400., -1.
    assert a.step('DOWN') == ['flight_crash']
    for _ in range(30):
        a.step('')
    assert not a.crash and a.episode > 1
    spawns = {(round(f.x), round(f.y), round(f.heading, 2), f.airborne) for f in [fl.Flight(s) for s in range(12)]}
    assert len(spawns) >= 4


def test_gentle_runway_landing_and_water_crash():
    from .apps.flight.world import AIRFIELDS, ISLAND_HEIGHT
    f = fl.Flight(0)
    cx, cy = AIRFIELDS[0]
    f.x, f.y, f.alt, f.heading, f.pitch, f.roll, f.speed, f.throttle, f.airborne = cx, cy - 100, ISLAND_HEIGHT + 2.3, 0., -.05, 0., 2.8, 0., True
    events = []
    for _ in range(5):
        events += f.step('')
    assert 'flight_land' in events and not f.crash and not f.airborne
    f.x, f.y, f.alt, f.airborne = 600., -600., 2.3, True
    assert f.step('DOWN') == ['flight_crash']


def test_flight_tiles_and_existing_model():
    s = MicroOS(3)
    empty = s.observe([])
    s.open_app('flight')
    obs = s.observe([])
    assert set(obs) == set(empty) == set(OBS_KEYS) | {'events'}
    tiles = fl.render_tiles(s.flight)
    assert tiles.shape == (19, 28)
    assert np.all(tiles < VOCAB)
    assert obs_to_pixels(obs).shape == (192, 256)
    assert np.all(obs['state'] < STATE_VOCAB)
    s.step((0, 0, 0, KEY_PLUS))
    c, r = s.pos['flight']
    assert np.array_equal(obs['grid'][r+1:r+20, c+1:c+29], tiles)
    torch.set_num_threads(1)
    model = WorldModel(Config(d=32, layers=1, heads=4, mem=2))
    assert not any(k.startswith('f_') for k in model.state_dict())
    for sample in [empty, obs]:
        x = {k: torch.tensor(sample[k].astype(np.int64))[None] for k in OBS_KEYS}
        x['grid'] = x['grid'].flatten(1)
        x.update(action=torch.zeros(1, 4, dtype=torch.long), prev_btn=torch.zeros(1, dtype=torch.long))
        out, _ = model(x, model.init_memory(1))
        loss, _ = compute_loss(out, x, x['grid'])
        assert torch.isfinite(loss)
        assert decode(out)['grid'].shape == (1, 768)
    s.open_app('paint')
    assert not np.array_equal(s.observe([])['grid'][r+1:r+20, c+1:c+29], tiles)


def test_edges_cockpit_clouds_and_instruments():
    for i, tile in enumerate(ART):
        chosen = match_tile(tile.tobytes())
        assert np.array_equal(ART[chosen], tile)
    f = _runway_start(0)
    clear = fl.render_tiles(f)
    assert clear[17, 8] == ATLAS[('flight_dial', 'speed', 0, 0, 0)]
    f.heading = next(b for b, _, _ in __import__('dreaming.apps.flight.world', fromlist=['clouds']).clouds(0))
    assert 7 in render_scene(f)[:45]
    f.roll = .6
    assert not np.array_equal(clear, fl.render_tiles(f))
    f.speed, f.alt, f.pitch = 6, 300, .5
    changed = fl.render_tiles(f)
    assert not np.array_equal(changed[17:19, 8:10], clear[17:19, 8:10])
    assert not np.array_equal(changed[17:19, 12:14], clear[17:19, 12:14])
    pixels = grid_to_pixels(changed)
    assert 7 in pixels[136:152]


def test_compact_atlas_and_shared_gauge_parts():
    from .flight_art import cockpit
    panel = cockpit()
    assert not np.any(panel == 11)
    assert np.all(panel[104:128, 112] == 255)
    assert np.all(panel[128:] != 255)
    ids = {value for key, value in ATLAS.ids.items() if key[0].startswith('flight_')}
    assert len(ids) <= 1024
    assert all(2048 <= value < 2304 or 2304 <= value < 3072 for value in ids)
    for level in range(16):
        assert ATLAS[('flight_dial', 'speed', level, 0, 0)] == ATLAS[('flight_dial', 'altitude', level, 0, 0)]
    assert len({ATLAS[('flight_dial', 'speed', level, r, c)]
                for level in range(16) for r in range(2) for c in range(2)}) < 70


def test_flight_and_existing_app_state_slots():
    assert [item[2][0] for item in DESK_ICONS[-4:]] == ['calculator', 'terminal', 'dreamer', 'flight']
    s = MicroOS(3)
    s.open_app('calculator')
    s.open_app('terminal')
    s.open_app('dreamer')
    before = s.ext_state_fields()
    s.open_app('flight')
    s.flight.alt, s.flight.speed, s.flight.throttle = 120, 3, .5
    after = s.ext_state_fields()
    assert after[:12] == before[:12]
    assert after[12:16] == [10, 15, 15, 0]


def test_banked_horizon_attitude_and_cracked_windscreen(monkeypatch):
    from .apps.flight import render
    monkeypatch.setattr(render, 'terrain', lambda x, y, seed: (np.zeros_like(x), np.zeros_like(x, dtype=int)))
    f = fl.Flight(0)
    f.x, f.y, f.alt, f.roll, f.pitch, f.crash = 5000., 5000., 80, .4, 0., 0
    scene = render_scene(f)
    horizon = np.argmax(scene == SEA, axis=0)
    assert horizon[208] - horizon[16] > 60
    banked = fl.render_tiles(f)
    f.roll = 0
    level = fl.render_tiles(f)
    assert not np.array_equal(banked[17:19, 12:14], level[17:19, 12:14])
    f.crash = 30
    crashed = fl.render_tiles(f)
    assert np.count_nonzero(crashed != level) >= 8
    assert crashed[5, 12] in {ATLAS[('flight_crack', color, 2)] for color in FRAME_COLORS}
    f.step('ENTER')
    assert not f.crash


def test_badly_matched_edges_are_not_flattened():
    patch = np.full((8, 8), SKY, np.uint8)
    patch[5:, :] = GRASS
    patch[4, 4:] = GRASS
    assert match_tile(patch.tobytes()) >= len(FILLS)


def test_flight_extended_palette_pixel_transport():
    from .tiles import PAL
    from .web.play import encode
    from .web.images import png
    from io import BytesIO
    from PIL import Image
    s = MicroOS(0)
    s.open_app('flight')
    ob = s.observe([])
    pixels = obs_to_pixels(ob)
    assert PAL.shape == (18, 3)
    assert pixels.max() < len(PAL)
    assert pixels.max() < len(PAL)
    encoded = encode([ob], False)
    assert encoded[2:] == pixels.astype(np.uint8).tobytes()
    assert encode([], False) == bytes([0, 0])
    image = np.asarray(Image.open(BytesIO(png(pixels).body)))
    assert np.array_equal(image, PAL[pixels])


def test_driver_opens_flight_with_its_icon_covered():
    from .driver import Driver
    for seed in (0, 7):
        s = MicroOS(seed)
        s.open_app('doom')
        d = Driver(s, seed)
        events = []
        for action in d.task_flight():
            obs = s.step(action)
            d.observe_events(obs['events'])
            events.extend(obs['events'])
        assert 'open_flight' in events
        assert 'flight_fly' in events


def test_simple_world_tile_budget_and_horizon_pairs():
    ids = {value for key, value in ATLAS.ids.items() if key[0].startswith('flight_')}
    assert len(ids) <= 1024
    assert (SKY, SEA) in PAIRS
    from .apps.flight.world import terrain
    h, m = terrain(np.array([5000., 0.]), np.array([5000., 0.]))
    assert m[0] == 0 and h[0] == 0 and m[1] in (4, 5)


def test_autopilot_takes_off_lands_and_crashes():
    from collections import Counter
    from .driver import Driver
    events = Counter()
    for seed in range(3):
        s = MicroOS(seed)
        d = Driver(s, seed, 'flight')
        for _ in range(2500):
            obs = s.step(next(d.stream))
            d.observe_events(obs['events'])
            events.update(obs['events'])
    assert events['flight_takeoff'] and events['flight_land'] and events['flight_crash']
