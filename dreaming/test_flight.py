import numpy as np
import pytest
from .apps import flight as fl
from .apps.flight.render import render_scene, match_tile, ART
from .os_sim import MicroOS, APPS, APP_SLOTS, STATE_VOCAB, DESK_ICONS
from .tiles import VOCAB, ATLAS, grid_to_pixels
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


def test_flight_tiles():
    tiles = fl.render_tiles(fl.Flight(3))
    assert tiles.shape == (19, 28)
    assert np.all(tiles < VOCAB)
    assert grid_to_pixels(tiles).shape == (152, 224)


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


def test_retired_flight_state_slots():
    assert len(APPS) == 8 and 'flight' not in APPS
    assert all(target[0] != 'flight' for _, _, target in DESK_ICONS)
    assert APP_SLOTS.index('flight') == 8
    s = MicroOS(3)
    for app in APPS:
        s.open_app(app)
    with pytest.raises(ValueError, match='Unknown app: flight'):
        s.open_app('flight')
    assert len(s.state_vector()) == len(STATE_VOCAB) == 78
    assert list(s.state_vector()[24:27]) == [0, 0, 0]
    assert s.ext_state_fields()[12:16] == [0, 0, 0, 0]


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


def test_flight_palette_pixel_transport():
    from .tiles import PAL
    from .web.images import png
    from io import BytesIO
    from PIL import Image
    pixels = grid_to_pixels(fl.render_tiles(fl.Flight(0)))
    assert PAL.shape == (16, 3)
    assert pixels.max() < len(PAL)
    image = np.asarray(Image.open(BytesIO(png(pixels).body)))
    assert np.array_equal(image, PAL[pixels])


def test_mixed_driver_never_opens_flight():
    from .driver import Driver
    s = MicroOS(7)
    d = Driver(s, 7, 'mixed')
    for _ in range(10000):
        obs = s.step(next(d.stream))
        d.observe_events(obs['events'])
        assert 'open_flight' not in obs['events']
        assert 'flight' not in s.z
        assert s.ext_state_fields()[12:16] == [0, 0, 0, 0]


def test_simple_world_tile_budget_and_horizon_pairs():
    ids = {value for key, value in ATLAS.ids.items() if key[0].startswith('flight_')}
    assert len(ids) <= 1024
    assert (SKY, SEA) in PAIRS
    from .apps.flight.world import terrain
    h, m = terrain(np.array([5000., 0.]), np.array([5000., 0.]))
    assert m[0] == 0 and h[0] == 0 and m[1] in (4, 5)


def test_standalone_endpoints():
    from fastapi.testclient import TestClient
    from .web.app import create_app
    from .web import flight
    from io import BytesIO
    from PIL import Image
    import base64
    with TestClient(create_app()) as client:
        page = client.get('/flight')
        assert page.status_code == 200 and 'text/html' in page.headers['content-type']
        assert 'simulator only - not part of the dreaming model' in page.text
        config = client.get('/config')
        assert config.status_code == 200
        assert set(config.json()) == {'pal', 'charset', 'extra_keys', 'key_extra0', 'sections', 'ckpts', 'current'}
        assert client.get('/sprites/flight').headers['content-type'] == 'image/png'
        a = client.post('/flight/api/step', json={'seed': 7}).json()
        b = client.post('/flight/api/step', json={'seed': 7}).json()
        for key in ['W'] * 5 + ['UP', 'RIGHT', '', 'ENTER']:
            a = client.post('/flight/api/step', json={'session': a['session'], 'key': key}).json()
            b = client.post('/flight/api/step', json={'session': b['session'], 'key': key}).json()
            assert {k: v for k, v in a.items() if k != 'session'} == {k: v for k, v in b.items() if k != 'session'}
        image = Image.open(BytesIO(base64.b64decode(a['frame'].split(',')[1])))
        assert image.size == (224, 152)
        assert client.post('/flight/api/step', json={'session': a['session'], 'reset': True}).json()['events'] == ['flight_reset']
        assert client.post('/flight/api/step', json={}).status_code == 200
        assert client.post('/flight/api/step', json={'session': 'missing'}).status_code == 404
        assert client.post('/flight/api/step', json={'key': 'invalid'}).status_code == 422
        assert client.post('/flight/api/step', json={'seed': -1}).status_code == 422
        for _ in range(flight.MAX_SESSIONS + 1):
            assert client.post('/flight/api/step', json={}).status_code == 200
        assert len(flight.sessions) == flight.MAX_SESSIONS
        assert client.post('/flight/api/step', json={'session': a['session']}).status_code == 404
