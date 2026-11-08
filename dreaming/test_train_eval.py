from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .eval import evaluate_mode, heldout_paths
from .model import Config, WorldModel
from .os_sim import STATE_VOCAB
from .tiles import ATLAS
from .train import MixedWindows, Windows, load_data, load_training_shards, to_torch, unroll_loss


def dataset():
    n = 40
    return dict(grid=np.full((n, 24, 32), ATLAS.used[0]), cursor=np.zeros((n, 2)),
                doom_cols=np.zeros((n, 160, 4)), doom_spr=np.zeros((n, 4, 3)),
                doom_meta=np.zeros((n, 3)), state=np.zeros((n, len(STATE_VOCAB))),
                action=np.zeros((n, 4)), ep_len=np.array([20, 20]),
                mode=np.array(['doom'] * 20 + ['windows'] * 20))


def test_sampler_mass_idle_and_modalities():
    d = dataset()
    d['doom_meta'][1:20, 0] = 1
    win = Windows(d, 2, np.random.default_rng(0), True)
    mask = d['mode'][win.valid] == 'doom'
    assert win.p[mask].sum() == pytest.approx(0.5)
    assert win.p[mask].min() / win.p[mask].max() >= 1 / 11
    explicit = Windows(d, 2, np.random.default_rng(0), True, {'doom': 3, 'windows': 1})
    assert explicit.p[mask].sum() == pytest.approx(0.75)
    assert win.p[0] > win.p[3]


@pytest.mark.parametrize('mix', [{'missing': 1}, {'doom': -1}, {'doom': 0}, {'doom': float('nan')}])
def test_invalid_mode_mix(mix):
    with pytest.raises(ValueError, match='mode mix'):
        Windows(dataset(), 2, np.random.default_rng(0), True, mix)


def test_empty_windows():
    with pytest.raises(ValueError, match='no episodes'):
        Windows(dataset(), 30, np.random.default_rng(0))


def test_mixed_batch():
    a = Windows(dataset(), 2, np.random.default_rng(0))
    d = dataset()
    d['cursor'][:] = 12
    b = Windows(d, 2, np.random.default_rng(1))
    batch = MixedWindows(a, b, 0.25).batch(8)
    assert (batch['cursor'][:, 0, 0] == 12).sum() == 2


def test_split_and_holdout_selection(tmp_path):
    d = dataset()
    d.pop('mode')
    path = tmp_path / 'shard_doom_001_00000.npz'
    np.savez(path, **d)
    train = load_training_shards([str(path)], holdout_episodes=1)
    val = load_training_shards([str(path)], holdout_episodes=1, heldout=True)
    assert len(train['action']) == len(val['action']) == 20
    assert train['mode'][0] == 'doom'
    assert heldout_paths(str(tmp_path)) == {'doom': str(path)}
    with pytest.raises(ValueError, match='no held-out'):
        heldout_paths(str(tmp_path), str(tmp_path))
    with pytest.raises(ValueError, match='at least one'):
        load_training_shards([str(path)], holdout_episodes=2)
    with pytest.raises(ValueError, match='no eligible'):
        load_training_shards([])


def test_extra_shard_cap(tmp_path):
    d = dataset()
    d.pop('mode')
    np.savez(tmp_path / 'shard_doom_001_00000.npz', **d)
    args = SimpleNamespace(data=str(tmp_path), max_steps_data=0, holdout_episodes=0,
                           sampler_v2=False, mode_mix=None, unroll=2, extra_data=str(tmp_path),
                           extra_frac=0.15, extra_shards=41)
    with pytest.raises(ValueError, match='shard cap'):
        load_data(args, 0, 1)


def test_sync_metrics_and_eval():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = WorldModel(Config(d=8, layers=1, heads=1, mem=2))
    d = dataset()
    b = to_torch(Windows(d, 1, np.random.default_rng(0)).batch(1), 'cpu')
    metrics = []
    for sync in (False, True):
        agg = {}
        loss = unroll_loss(model, model, b, SimpleNamespace(unroll=1, sync_fix=sync),
                           0, 0, torch.Generator().manual_seed(0), 'cpu', agg)
        assert torch.isfinite(loss)
        metrics.append({k: float(v) for k, v in agg.items()})
    assert metrics[0] == metrics[1]
    result = evaluate_mode(model, d, 'cpu', windows=2, batch=1, starts=1, horizons=(2,))
    assert np.isfinite(result['grid_ce'])
    assert result['copy_last_acc_all'] == 1
    assert result['acc_changed'] == 0
    assert len(result['rollouts']['2']['acc_per_step']) == 2
    with pytest.raises(ValueError, match='positive'):
        evaluate_mode(model, d, 'cpu', windows=0)
