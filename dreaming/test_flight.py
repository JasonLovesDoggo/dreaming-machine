import numpy as np
import torch
from .apps import flight as fl
from .apps.flight.render import render_scene, match_tile, ART
from .os_sim import MicroOS, obs_to_pixels, STATE_VOCAB, KEY_PLUS, DESK_ICONS
from .tiles import VOCAB, ATLAS, grid_to_pixels
from .model import WorldModel, Config, compute_loss, decode
from .dataset import OBS_KEYS
from .flight_art import SKY, HAZE, SEA, GRASS, FRAME_COLORS, FILLS


def test_flight_takeoff_crash_reset_and_determinism():
    a, b = fl.Flight(3), fl.Flight(3)
    for key in ['W'] * 30 + [''] * 25 + ['UP'] * 14 + [''] * 60:
        assert a.step(key) == b.step(key)
    assert a.alt > 20
    assert np.array_equal(fl.render_tiles(a), fl.render_tiles(b))
    a.x, a.y, a.alt = 400., 400., -1.
    assert a.step('DOWN') == ['flight_crash']
    for _ in range(30):
        a.step('')
    assert a.alt == 10 and a.speed == 0


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
    f = fl.Flight(0)
    clear = fl.render_tiles(f)
    assert clear[17, 8] == ATLAS[('flight_dial', 'speed', 0, 0, 0)]
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
    f.alt, f.roll = 80, .4
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


def test_perlin_world_continuity_seed_and_flat_airfields():
    from .apps.flight.world import fbm, perlin, terrain
    x, y = np.meshgrid(np.linspace(-4, 4, 32), np.linspace(-3, 3, 24))
    a = fbm(x, y, 7)
    assert np.array_equal(a, fbm(x, y, 7))
    assert not np.array_equal(a, fbm(x, y, 8))
    assert np.isfinite(a).all() and np.max(np.abs(a)) < 1
    assert np.max(np.abs(perlin(x + 1e-6, y, 7) - perlin(x, y, 7))) < 1e-4
    assert fbm(np.array([]), np.array([]), 7).size == 0
    for cx, cy in ((0, 0), (1200, 900)):
        height, material = terrain(cx + np.array([-10, 0, 10]), cy, 7)
        assert np.all(height == 8) and np.all(np.isin(material, (4, 5)))


def test_badly_matched_edges_are_not_flattened():
    patch = np.full((8, 8), SKY, np.uint8)
    patch[5:, :] = GRASS
    patch[4, 4:] = GRASS
    assert match_tile(patch.tobytes()) >= len(FILLS)


def test_every_two_colour_boundary_has_a_matching_non_solid_tile():
    from itertools import combinations
    from .apps.flight.render import EDGE_END
    for a, b in combinations(FILLS, 2):
        for count in (1, 7, 17, 32, 55, 63):
            patch = np.full((8, 8), a, np.uint8)
            patch.flat[:count] = b
            index = match_tile(patch.tobytes())
            assert len(FILLS) <= index < EDGE_END
            chosen = ART[index]
            assert set(np.unique(chosen)) == {a, b}


def test_sparse_clouds_haze_and_muted_palette(monkeypatch):
    from .apps.flight import render
    monkeypatch.setattr(render, 'terrain', lambda x, y, seed: (np.zeros_like(x), np.zeros_like(x, dtype=int)))
    for seed in (0, 7):
        f = fl.Flight(seed)
        f.alt = 80
        scene = render_scene(f)
        sky = render.YY < 53
        coverage = np.count_nonzero((scene == 7) & sky) / sky.sum()
        assert .05 <= coverage <= .15
        assert not np.any((scene == 7) & ~sky)
        assert HAZE in scene[45:58]
        assert 11 not in scene
    assert match_tile(np.full((8, 8), SEA, np.uint8).tobytes()) < len(FILLS)


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
    assert np.any(pixels >= 16)
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
