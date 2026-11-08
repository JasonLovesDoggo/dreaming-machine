import numpy as np
import torch
from .apps import flight as fl
from .apps.flight.render import render_scene, match_tile, ART
from .os_sim import MicroOS, obs_to_pixels, STATE_VOCAB
from .tiles import VOCAB, ATLAS, grid_to_pixels
from .model import WorldModel, Config, compute_loss, decode
from .dataset import OBS_KEYS


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
    assert clear[14, 8] == ATLAS[('flight_dial', 'speed', 0, 0, 0)]
    assert 7 in render_scene(f)[:45]
    f.roll = .6
    assert not np.array_equal(clear, fl.render_tiles(f))
    f.speed, f.alt, f.pitch = 6, 300, .5
    changed = fl.render_tiles(f)
    assert not np.array_equal(changed[14:17, 8:11], clear[14:17, 8:11])
    assert not np.array_equal(changed[14:17, 12:15], clear[14:17, 12:15])
    pixels = grid_to_pixels(changed)
    assert 7 in pixels[112:136]


def test_compact_atlas_and_shared_gauge_parts():
    ids = {value for key, value in ATLAS.ids.items() if key[0].startswith('flight_')}
    assert len(ids) <= 512
    assert all(2048 <= value < 2304 or 2304 <= value < 2560 for value in ids)
    for level in range(16):
        assert ATLAS[('flight_dial', 'speed', level, 0, 0)] == ATLAS[('flight_dial', 'altitude', level, 0, 0)]
    assert len({ATLAS[('flight_dial', 'speed', level, r, c)]
                for level in range(16) for r in range(3) for c in range(3)}) < 70


def test_banked_horizon_attitude_and_cracked_windscreen(monkeypatch):
    from .apps.flight import render
    monkeypatch.setattr(render, 'terrain', lambda x, y, seed: (np.zeros_like(x), np.zeros_like(x, dtype=int)))
    f = fl.Flight(0)
    f.alt, f.roll = 80, .4
    scene = render_scene(f)
    horizon = np.argmax(scene == 1, axis=0)
    assert horizon[208] - horizon[16] > 60
    banked = fl.render_tiles(f)
    f.roll = 0
    level = fl.render_tiles(f)
    assert not np.array_equal(banked[14:17, 12:15], level[14:17, 12:15])
    f.crash = 30
    crashed = fl.render_tiles(f)
    assert np.count_nonzero(crashed != level) >= 8
    assert crashed[5, 12] in {ATLAS[('flight_crack', color, 2)] for color in (12, 1, 14, 13)}
    f.step('ENTER')
    assert not f.crash
