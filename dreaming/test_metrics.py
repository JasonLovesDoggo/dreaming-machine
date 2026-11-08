import json

import numpy as np
import pytest
import torch

from . import eval as evaluation, metrics, train
from .model import Config, WorldModel
from .test_train_eval import dataset


def test_append_and_nonfinite(tmp_path):
    out = tmp_path / 'new'
    metrics.append_metrics(out, {'loss': float('nan'), 'nested': [float('inf'), {'x': -float('inf')}]})
    metrics.append_metrics(out, {'it': 2})
    rows = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert rows == [{'loss': None, 'nested': [None, {'x': None}]}, {'it': 2}]


def test_io_failure_warns_once(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, '_warned', False)
    def fail(*args, **kwargs):
        raise OSError('test disk failure')
    monkeypatch.setattr(metrics, 'open', fail, raising=False)
    with pytest.warns(RuntimeWarning, match='test disk failure') as warnings:
        metrics.append_metrics(tmp_path, {})
        metrics.append_metrics(tmp_path, {})
    assert len(warnings) == 1


def test_train_loop_metrics(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    win = train.Windows(dataset(), 1, np.random.default_rng(0))
    monkeypatch.setattr(train, 'load_data', lambda *args: win)
    monkeypatch.setattr(train, 'save_checkpoint', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('sys.argv', ['train', '--out', str(tmp_path), '--iters', '2', '--batch', '1',
                                    '--unroll', '1', '--d', '8', '--layers', '1', '--heads', '1',
                                    '--mem', '2', '--log-every', '1', '--sync-fix'])
    train.main()
    rows = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert [r['it'] for r in rows] == [1, 2]
    assert all(r['kind'] == 'train' for r in rows)
    assert set(rows[0]) == {'kind', 'it', 'time', 'loss', 'grid', 'cursor', 'doom', 'state',
                            'acc_all', 'acc_changed', 'frac_changed', 'lr', 'grad_norm',
                            'ss_prob', 'samples_per_sec'}


@pytest.mark.parametrize('override', [False, True])
def test_eval_metrics(tmp_path, monkeypatch, override):
    torch.set_num_threads(1)
    d = dataset()
    d.pop('mode')
    np.savez(tmp_path / 'shard_doom_001_00000.npz', **d)
    model = WorldModel(Config(d=8, layers=1, heads=1, mem=2))
    ck = tmp_path / 'ckpt.pt'
    torch.save(dict(model=model.state_dict(), cfg=model.cfg.__dict__, iter=17), ck)
    out = tmp_path / 'results.json'
    metric_dir = tmp_path / 'override' if override else tmp_path
    args = ['eval', '--checkpoint', str(ck), '--data', str(tmp_path), '--out', str(out),
            '--windows', '1', '--starts', '1', '--horizons', '2']
    if override:
        args.extend(['--metrics-dir', str(metric_dir)])
    monkeypatch.setattr('sys.argv', args)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    evaluation.main()
    rows = [json.loads(line) for line in (metric_dir / 'metrics.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['kind'] == 'eval'
    assert rows[0]['it'] == 17
    assert rows[0]['checkpoint'] == 'ckpt.pt'
    assert rows[0]['metrics'] == json.loads(out.read_text())
