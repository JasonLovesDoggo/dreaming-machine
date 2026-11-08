from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .eval import HELDOUT_SEED_BASE, changed_transition_counts, evaluate_mode, false_change_counts, heldout_paths
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


def test_mode_mix_only_applies_to_primary(tmp_path):
    primary, extra = tmp_path / 'primary', tmp_path / 'extra'
    primary.mkdir()
    extra.mkdir()
    d = dataset()
    d.pop('mode')
    np.savez(primary / 'shard_mixed_001_00000.npz', **d)
    for i in range(2):
        np.savez(extra / f'shard_cats_001_{i:05d}.npz', **d)
    args = SimpleNamespace(data=str(primary), max_steps_data=0, holdout_episodes=0,
                           sampler_v2=True, mode_mix={'mixed': 1}, unroll=2,
                           extra_data=str(extra), extra_frac=0.25, extra_shards=1)
    win = load_data(args, 0, 1)
    assert isinstance(win, MixedWindows)
    assert set(win.extra.d['mode']) == {'cats'}
    assert win.batch(4)['grid'].shape[0] == 4


def test_reserved_heldout_seeds():
    heldout = {((HELDOUT_SEED_BASE + i) * 10_000_000 + e) for i in range(7) for e in range(2)}
    training = {base * 10_000_000 + shard * 1000 + e
                for base in range(1000, 3000) for shard in (0, 999) for e in (0, 24)}
    assert heldout.isdisjoint(training)
    assert min(heldout) > max(training)


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


def test_false_change_counts():
    previous = torch.tensor([1, 2, 3, 4])
    truth = torch.tensor([1, 2, 8, 4])
    predicted = torch.tensor([1, 9, 7, 4])
    assert false_change_counts(predicted, truth, previous) == (1, 3)
    assert false_change_counts(previous, truth, previous) == (0, 3)
    assert false_change_counts(previous, previous + 1, previous) == (0, 0)


def test_changed_transition_sizes():
    previous = torch.zeros(3, 20, dtype=torch.long)
    truth = previous.clone()
    truth[0, :2] = 1
    truth[1] = 1
    predicted = previous.clone()
    predicted[0, 0] = 1
    predicted[1, :5] = 1
    assert changed_transition_counts(predicted, truth, previous) == dict(
        small_correct=1, small_tiles=2, small_frames=1,
        large_correct=5, large_tiles=20, large_frames=1)


def test_rollout_false_changes_steps_and_frame_dump(tmp_path):
    from PIL import Image
    torch.set_num_threads(1)
    d = dataset()
    d = {k: np.repeat(v[:1], 140, axis=0) for k, v in d.items() if k != 'ep_len'}
    d['ep_len'] = np.array([140])
    real = WorldModel(Config(d=8, layers=1, heads=1, mem=2)).eval()
    b = to_torch(Windows(d, 1, np.random.default_rng(0)).batch(1), 'cpu')
    with torch.no_grad():
        out, _ = real({k: v[:, 0] for k, v in b.items()}, real.init_memory(1))
    out['grid'] = torch.full_like(out['grid'], -10)
    out['grid'][..., ATLAS.used[1]] = 10
    out['m_gun'][:, 0] = 100
    class FixedPredictions:
        def init_memory(self, batch):
            return real.init_memory(batch)

        def __call__(self, x, mem):
            return out, mem
    result = evaluate_mode(FixedPredictions(), d, 'cpu', windows=1, batch=1, starts=1,
                           horizons=(100,), dump_frames=str(tmp_path))
    assert result['fp_change_rate'] == 1
    assert result['copy_last_fp_change_rate'] == 0
    assert result['acc_changed_small'] == result['acc_changed_large'] == 0
    assert result['changed_small_frames'] == result['changed_large_frames'] == 0
    rollout = result['rollouts']['100']
    assert rollout['fp_change_rate_per_step'] == [1.0] * 100
    assert rollout['copy_last_fp_change_rate_per_step'] == [0.0] * 100
    assert rollout['acc_at_steps'] == {'10': 0, '50': 0, '100': 0}
    assert rollout['copy_first_acc_at_steps'] == {'10': 1, '50': 1, '100': 1}
    with Image.open(tmp_path / 'start_00.png') as image:
        assert image.size == (512, 4 * (192 + 18))
