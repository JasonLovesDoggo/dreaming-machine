import json

import numpy as np
import pytest

from .dataset import OBS_KEYS, load_shards
from .picture_pilot import (CORE_NAMES, SEED_BASES, SEED_SPAN, episode_seed, filename_for_seed,
                            generate, replay_matches, run_episode)


def test_split_seed_namespaces_are_disjoint():
    train = range(SEED_BASES['train'], SEED_BASES['train'] + SEED_SPAN)
    heldout = range(SEED_BASES['heldout'], SEED_BASES['heldout'] + SEED_SPAN)
    assert train.stop <= heldout.start
    assert episode_seed('train', 0) == 200_000_000
    assert episode_seed('heldout', 0) == 210_000_000


@pytest.mark.parametrize('split,index', [('bad', 0), ('train', -1), ('train', SEED_SPAN)])
def test_invalid_seed_namespace(split, index):
    with pytest.raises(ValueError, match='seed index'):
        episode_seed(split, index)


@pytest.mark.parametrize('filename', CORE_NAMES)
@pytest.mark.parametrize('index', [0, 697])
def test_complete_load_and_exact_replay(filename, index):
    data, meta, _ = run_episode(episode_seed('train', index), filename)
    assert set(data) == set(OBS_KEYS + ['action'])
    assert meta['load_stages'] == list(range(1, 11))
    assert meta['complete_tick'] - meta['open_tick'] == 9
    assert meta['ticks'] - meta['complete_tick'] >= 12
    assert meta['counts']['newly_revealed_cells'] == 160
    assert meta['counts']['loading_ticks'] == 9
    replay_matches(data, meta)
    repeated, repeated_meta, _ = run_episode(episode_seed('train', index), filename)
    assert repeated_meta == meta
    for key in data:
        np.testing.assert_array_equal(data[key], repeated[key])


def test_hidden_seed_variation_and_broader_names():
    pictures = [run_episode(episode_seed('train', i), 'ORANGE CAT.PNG')[2] for i in range(4)]
    assert len({p.tobytes() for p in pictures}) > 1
    names = {filename_for_seed(episode_seed('train', i)) for i in range(512)}
    assert set(CORE_NAMES) <= names
    assert len(names) > 100


def test_corrupt_replay_is_rejected():
    data, meta, _ = run_episode(episode_seed('train', 0))
    data['grid'][-1, 0, 0] ^= 1
    with pytest.raises(ValueError, match='replay mismatch'):
        replay_matches(data, meta)


def test_corrupt_action_hash_and_canvas_are_rejected():
    data, meta, _ = run_episode(episode_seed('train', 0))
    with pytest.raises(ValueError, match='action hash'):
        replay_matches(data, dict(meta, action_sha256='bad'))
    with pytest.raises(ValueError, match='current generator'):
        replay_matches(data, dict(meta, completed_canvas_sha256='bad'))


@pytest.mark.parametrize('name', ['NOTES.TXT', 'TOO_LONG_' * 4 + '.PNG', 'BAD?.PNG'])
def test_unsupported_name_is_rejected(name):
    with pytest.raises(ValueError, match='filename'):
        run_episode(episode_seed('train', 0), name)


def test_shards_match_dataset_format_and_sidecars(tmp_path):
    summary = generate(str(tmp_path), 200, 'heldout', workers=1)
    paths = sorted(tmp_path.glob('*.npz'))
    data = load_shards(paths)
    assert set(data) == set(OBS_KEYS + ['action', 'ep_len'])
    assert sum(data['ep_len']) == len(data['action'])
    assert len(data['action']) - len(data['ep_len']) == summary['ticks'] >= 200
    sidecar = json.loads(paths[0].with_suffix('.npz.json').read_text())
    assert sidecar['mode'] == 'picture_pilot' and sidecar['split'] == 'heldout'
    assert len(sidecar['source_sha']) == 40
    assert {'os_sim.py', 'driver.py', 'pictures/__init__.py', 'pictures/cats.py'} <= set(sidecar['source_hashes'])
    start = 0
    for length, meta in zip(data['ep_len'], sidecar['episodes'], strict=True):
        replay_matches({k: data[k][start:start + length] for k in OBS_KEYS + ['action']}, meta)
        start += length
    assert summary['validated_episodes'] == summary['episodes']
    assert (tmp_path / 'contact_sheet.png').exists()
    with pytest.raises(ValueError, match='fresh output'):
        generate(str(tmp_path), 1, 'heldout')


@pytest.mark.parametrize('steps,workers', [(0, 1), (-1, 1), (10, 0), (10, 17)])
def test_invalid_generation_limits(tmp_path, steps, workers):
    with pytest.raises(ValueError, match='positive'):
        generate(str(tmp_path), steps, 'train', workers=workers)
