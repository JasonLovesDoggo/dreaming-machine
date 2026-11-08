import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from . import train
from .cluster import check_lock, config
from .model import Config, WorldModel
from .picture_preflight import (check_sidecar, fingerprint, inspect_pictures, loader_args, loader_probe,
                                manifest_summaries, model_bounds, model_step, optimizer_report,
                                run_preflight, sandbox, shard_stats)
from .test_train_eval import dataset


@pytest.fixture
def data():
    result = dataset()
    result.pop('mode')
    return {key: value.astype(np.int64) for key, value in result.items()}


@pytest.fixture
def model():
    torch.set_num_threads(1)
    return WorldModel(Config(d=16, layers=1, heads=2, mem=2))


def test_shard_validation(data, model):
    result = shard_stats(data, data, model_bounds(model))
    assert (result['frames'], result['ticks'], result['episodes']) == (40, 38, 2)
    for key, value in [('grid', model.tile.num_embeddings), ('action', -1)]:
        bad = copy.deepcopy(data)
        bad[key].flat[0] = value
        with pytest.raises(ValueError, match='range violation'):
            shard_stats(bad, data, model_bounds(model))
    for change, match in [('keys', 'keys'), ('length', 'ep_len'), ('dtype', 'dtype'), ('shape', 'shape')]:
        bad = copy.deepcopy(data)
        if change == 'keys':
            bad['unexpected'] = np.zeros(1)
        elif change == 'length':
            bad['ep_len'][0] += 1
        elif change == 'dtype':
            bad['cursor'] = bad['cursor'].astype(np.uint8)
        else:
            bad['cursor'] = bad['cursor'][:, :1]
        with pytest.raises(ValueError, match=match):
            shard_stats(bad, data, model_bounds(model))


def test_fingerprint(tmp_path):
    path = tmp_path / 'ckpt_024000.pt'
    path.write_bytes(b'immutable')
    before = fingerprint(path)
    assert fingerprint(path) == before
    path.write_bytes(b'changed')
    assert fingerprint(path) != before


def metadata_for(data):
    return dict(split='train', seeds=[200000000, 200000001], episodes=[
        dict(seed=200000000 + i, ticks=19, action_sha256=hashlib.sha256(data['action'][i * 20:(i + 1) * 20].tobytes()).hexdigest())
        for i in range(2)])


def test_sidecar(data, tmp_path):
    path = tmp_path / 'shard.npz'
    np.savez(path, **data)
    metadata = metadata_for(data)
    metadata['shard_sha256'] = fingerprint(path)['sha256']
    assert check_sidecar(path, data, metadata, {210000000}) == metadata['seeds']
    for key, value, match in [('split', 'heldout', 'non-train'), ('seeds', [210000000, 210000001], 'out of range'),
                              ('shard_sha256', 'bad', 'sha256')]:
        with pytest.raises(ValueError, match=match):
            check_sidecar(path, data, {**metadata, key: value}, {210000000})
    with pytest.raises(ValueError, match='overlap'):
        check_sidecar(path, data, metadata, {200000000})
    metadata['episodes'][0]['action_sha256'] = 'bad'
    with pytest.raises(ValueError, match='action sha256'):
        check_sidecar(path, data, metadata, set())


def test_manifest_schema_and_empty(tmp_path, data, model):
    manifest = dict(train=dict(parts=[dict(seeds=[200000000])]), heldout=dict(seeds=[210000000]))
    assert manifest_summaries(manifest) == [('train', dict(seeds=[200000000])), ('heldout', dict(seeds=[210000000]))]
    (tmp_path / 'manifest.json').write_text('{}')
    report = {}
    with pytest.raises(ValueError, match='lacks identifiable'):
        inspect_pictures(tmp_path, data, model, report)
    assert report['manifest'] == {}


def test_real_loader_and_step(tmp_path, data, model):
    primary, cats, pictures = tmp_path / 'primary', tmp_path / 'cats', tmp_path / 'train'
    for root in (primary, cats, pictures):
        root.mkdir()
    for i in range(24):
        np.savez(primary / f'shard_{i:03d}.npz', **data)
        np.savez(cats / f'shard_cats_{i:03d}.npz', **data)
    np.savez(pictures / 'picture.npz', **data)
    lock = check_lock(config('preflight-test', extra_shards=400, data=str(primary), extra_data=str(cats), holdout_episodes=0), None)
    args = loader_args(lock)
    assert (args.batch, args.unroll, args.extra_shards, args.max_steps_data) == (32, 6, 400, 20000)
    original = train.load_data
    small, result = loader_probe(args, str(pictures))
    assert train.load_data is original
    assert result['counts'] == dict(primary=22, cats=5, pictures=5)
    assert small['grid'].shape[:2] == (3, 2)
    optimizer = train.build_optimizer(model, args, 'cpu')
    batch = train.to_torch(small, 'cpu')
    args.unroll = 1
    loss = train.unroll_loss(model, model, batch, args, 0., 0., torch.Generator(), 'cpu', {})
    loss.backward()
    optimizer.step()
    ck = dict(opt=optimizer.state_dict())
    assert optimizer_report(ck['opt'])['step'] == 1
    result = model_step(model, ck, args, small)
    assert result['changed_parameters'] > 0 and result['finite']
    assert result['optimizer_after']['step'] == 2
    first = next(iter(ck['opt']['state'].values()))
    first['step'] = torch.tensor(float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        optimizer_report(ck['opt'])


def test_sandbox_does_not_change_source():
    now = 1794132000.  # 2026-11-08 10:00 UTC
    ledger = [dict(start=now - 3600, deadline=now + 7 * 3600)]
    before = copy.deepcopy(ledger)
    lock = check_lock(config('source-test', extra_shards=400), None)
    report = sandbox(lock, dict(iter=24000), ledger, {'ckpt_024000.pt'}, now)
    assert ledger == before and 'end' not in ledger[-1]
    assert report['used_h'] == 1.25 and report['remaining_cap_h'] == 6.75
    assert '--extra-shards 400' in report['torchrun_command']
    assert '--resume /src/runs/overnight_20261108_b30024/ckpt_024000.pt' in report['torchrun_command']
    assert report['cases'][0]['kill_at_utc'].endswith('14:19:00+00:00')
    assert report['cases'][1]['kill_at_utc'].endswith('14:29:18+00:00')
    ledger[0]['start'] = now - 7 * 3600
    report = sandbox(lock, dict(iter=24000), ledger, {'ckpt_024000.pt'}, now)
    assert report['cases'][0]['budget_clamps_earlier']
    ledger[0]['start'] = now - 9 * 3600
    report = sandbox(lock, dict(iter=24000), ledger, {'ckpt_024000.pt'}, now)
    assert 'refused' in report['cases'][0]


def test_report_on_missing_source(tmp_path):
    report = run_preflight(source=str(tmp_path))
    assert report['status'] == 'failed' and 'FileNotFoundError' in report['error']
    assert sorted(Path(tmp_path).iterdir()) == []
