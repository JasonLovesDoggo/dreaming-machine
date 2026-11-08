import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dreaming.cluster import (CalibrationGuard, TrainingGuardError, check_budget, config,
                              lock_config, merge_metrics, resume_path,
                              publish_checkpoints, publish_set, run_with_deadline,
                              torchrun_command, validate_run_id)
from dreaming.test_train_eval import dataset
from dreaming.warm_restart import restart_lr


def test_schedule():
    a = SimpleNamespace(lr=1., iters=1000000)
    assert restart_lr(5000, a, 5000, 200, .1) == .005
    assert restart_lr(5199, a, 5000, 200, .1) == pytest.approx(1., abs=1e-6)
    assert restart_lr(a.iters, a, 5000, 200, .1) == pytest.approx(.1)
    with pytest.raises(ValueError, match='not resumed'):
        restart_lr(4999, a, 5000, 200, .1)
    for r, w, m in [(a.iters, 200, .1), (5000, 0, .1), (5000, 1, 2)]:
        with pytest.raises(ValueError):
            restart_lr(5000, a, r, w, m)


def test_cpu_restart_end_to_end(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    np.savez(data / 'shard_mixed_001_00000.npz', **dataset())
    source, target = tmp_path / 'source', tmp_path / 'target'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    args = ['--data', str(data), '--iters', '12', '--batch', '1', '--unroll', '1',
            '--d', '8', '--layers', '1', '--heads', '1', '--mem', '2', '--log-every', '1']
    subprocess.run([sys.executable, '-m', 'dreaming.train', *args, '--out', str(source),
                    '--chunk', '3'], env=env, check=True, capture_output=True, timeout=120)
    ckpt = source / 'ckpt.pt'
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    result = subprocess.run([sys.executable, '-m', 'dreaming.warm_restart',
                            '--restart-iter', '3', '--restart-warmup', '2', '--min-lr-frac', '.1',
                            '--', *args, '--out', str(target), '--resume', str(ckpt), '--chunk', '3'],
                            env=env, check=True, capture_output=True, text=True, timeout=120)
    assert 'resumed at iter 3' in result.stdout
    rows = [json.loads(line) for line in (target / 'metrics.jsonl').read_text().splitlines()]
    assert [r['it'] for r in rows] == [4, 5, 6]
    for row in rows:
        assert row['lr'] == pytest.approx(restart_lr(row['it'] - 1, SimpleNamespace(lr=4e-4, iters=12), 3, 2, .1))
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == digest
    failed = subprocess.run([sys.executable, '-m', 'dreaming.warm_restart',
                             '--restart-iter', '3', '--restart-warmup', '2', '--min-lr-frac', '.1',
                             '--', *args, '--out', str(tmp_path / 'missing')], env=env,
                            capture_output=True, text=True, timeout=120)
    assert failed.returncode != 0 and 'checkpoint was not resumed' in failed.stderr


@pytest.mark.parametrize('run', ['main', 'atlas_v2_first_5k', 'legacy_v1', 'tput1', 'prof64',
                                 'prof32', 'audit_ab_A', 'audit_ab_E', '../other', 'ab', '', 'Aaa', 'a' * 49])
def test_invalid_run(run):
    with pytest.raises(ValueError):
        validate_run_id(run)


def test_config_lock_budget_and_command(tmp_path):
    cfg = config('pilot-test')
    assert validate_run_id('pilot-test') == 'dreaming-run-pilot-test'
    assert cfg['steps'] == 5000 and cfg['log_every'] == 10 and cfg['ckpt_every'] == 5000
    assert config('pilot-test', steps=5000)['end_iter'] == 10000
    with pytest.raises(ValueError):
        config('pilot-test', steps=1094)
    lock_config(tmp_path, cfg)
    lock_config(tmp_path, config('pilot-test', steps=5000))
    with pytest.raises(ValueError, match='mismatch'):
        lock_config(tmp_path, dict(cfg, lr=.01))
    with pytest.raises(ValueError, match='mismatch'):
        lock_config(tmp_path, dict(cfg, sampler_v2=True))
    assert check_budget([], 3600) == 0
    records = [dict(start=100, deadline=200), dict(start=200, deadline=300, end=250)]
    assert check_budget(records, 100, 250) == 150
    with pytest.raises(ValueError):
        check_budget(records, 101, 250)
    with pytest.raises(ValueError):
        check_budget([], 8 * 3600 + 1)
    assert resume_path(tmp_path) == cfg['source']
    (tmp_path / 'ckpt.pt').touch()
    assert resume_path(tmp_path) == 'auto'
    cmd = torchrun_command(cfg, 2, '10.0.0.1', resume='auto')
    assert '--node-rank=2' in cmd and '--master-addr=10.0.0.1' in cmd
    assert '--max-restarts=0' in cmd and '--nnodes=3' in cmd and '--nproc-per-node=8' in cmd
    assert cmd[cmd.index('-m') + 1] == 'dreaming.warm_restart'
    assert cmd[cmd.index('--resume') + 1] == 'auto'
    assert cmd[cmd.index('--chunk') + 1] == '5000'
    assert '--sync-fix' in cmd and '--compile' in cmd and '--sampler-v2' not in cmd
    assert cfg['sampler_v2'] is False
    assert '--sampler-v2' in torchrun_command(config('pilot-test', sampler_v2=True))


@pytest.mark.parametrize('cadence', [0, -1000, '1000', 1000.0, True])
def test_invalid_checkpoint_cadence(cadence):
    with pytest.raises(ValueError, match='positive int'):
        config('cadence-test', ckpt_every=cadence)


def test_checkpoint_cadence(tmp_path):
    default = config('cadence-test')
    assert default['ckpt_every'] == 5000
    assert default == config('cadence-test', ckpt_every=5000)
    assert torchrun_command(default) == torchrun_command(config('cadence-test', ckpt_every=5000))
    command = torchrun_command(default)
    assert command[command.index('--ckpt-every') + 1] == '5000'
    cfg = config('cadence-test', ckpt_every=1000, end_iter=7000)
    command = torchrun_command(cfg)
    assert command[command.index('--ckpt-every') + 1] == '1000'
    assert publish_set(cfg, 5000) == {'ckpt_006000.pt', 'ckpt_007000.pt'}
    assert config('cadence-test', ckpt_every=1000, end_iter=6000)['end_iter'] == 6000
    with pytest.raises(ValueError, match='multiple of ckpt_every'):
        config('cadence-test', ckpt_every=1000, end_iter=10500)
    with pytest.raises(ValueError, match='cosine horizon'):
        config('cadence-test', ckpt_every=1000, end_iter=66000)
    lock_config(tmp_path / 'out', default)
    lock_config(tmp_path / 'out', cfg)
    src, dst = tmp_path / 'out', tmp_path / 'cadence-test'
    for name in publish_set(cfg, 5000):
        (src / name).write_bytes(b'checkpoint')
    (src / 'run.json').write_text(json.dumps(dict(atlas_version=2, iter=6000)))
    publish_checkpoints(src, dst, ['ckpt_006000.pt'])
    with pytest.raises(ValueError, match='not complete'):
        publish_checkpoints(src, dst, ['ckpt_007000.pt'])
    (src / 'run.json').write_text(json.dumps(dict(atlas_version=2, iter=7000)))
    publish_checkpoints(src, dst, sorted(publish_set(cfg, 5000)))
    assert json.loads((dst / 'run.json').read_text())['iter'] == 7000


def test_resumed_publication_sets():
    cfg = config('long-test')
    assert publish_set(cfg, 5000) == {'ckpt_010000.pt'}
    assert config('long-test', steps=35000)['end_iter'] == 40000
    assert publish_set(config('long-test', end_iter=45000), 5000) == {
        f'ckpt_{it:06d}.pt' for it in range(10000, 45001, 5000)}
    resumed = config('long-test', start_iter=15000, end_iter=25000)
    assert resumed['steps'] == 10000
    assert publish_set(resumed, 15000) == {'ckpt_020000.pt', 'ckpt_025000.pt'}
    with pytest.raises(ValueError):
        config('long-test', phase='pilot')


def test_calibration_guard(tmp_path, capsys, caplog):
    path = tmp_path / 'metrics.jsonl'
    cfg = config('guard-test')
    guard = CalibrationGuard(path, cfg, time.time() + 1)
    guard()
    assert not guard.summarized
    path.write_text('')
    guard()
    row = dict(kind='train', it=5300, loss=1., grad_norm=.5, samples_per_sec=4608.)
    path.write_text(json.dumps(row) + '\n')
    guard()
    assert guard.summarized
    assert 'updates/s=1.0000' in capsys.readouterr().out
    assert 'warning only' in caplog.text
    guard()
    assert capsys.readouterr().out == ''
    for key, value in [('loss', None), ('grad_norm', float('nan')), ('loss', 2.1)]:
        path.write_text(json.dumps(dict(row, **{key: value})) + '\n')
        with pytest.raises(TrainingGuardError):
            guard()
    path.write_text(json.dumps(dict(row, it=5100, loss=3.)) + '\n')
    guard()


def test_guard_failure_kills_child():
    def fatal():
        raise TrainingGuardError('unsafe loss')

    with pytest.raises(TrainingGuardError, match='unsafe loss'):
        run_with_deadline([sys.executable, '-c', 'import time; time.sleep(60)'],
                          time.time() + 10, on_tick=fatal, tick_every=.1, grace=.2)


def test_dry_run_does_not_launch(monkeypatch, capsys):
    monkeypatch.setenv('DREAMING_RUN_ID', 'dry-test')
    import cluster_run
    monkeypatch.setattr(cluster_run, 'read_state', lambda: (None, [], False, None))

    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run must not call a remote function')

    monkeypatch.setattr(cluster_run.train_cluster, 'remote', forbidden)
    cluster_run.launch(dry_run=True)
    output = capsys.readouterr().out
    assert '"end_iter": 10000' in output and '--ckpt-every 5000' in output
    cluster_run.launch(end_iter=6000, ckpt_every=1000, dry_run=True)
    output = capsys.readouterr().out
    assert '"end_iter": 6000' in output and '--ckpt-every 1000' in output


def test_launch_resumes_completed_numbered_checkpoint(monkeypatch, capsys):
    monkeypatch.setenv('DREAMING_RUN_ID', 'resume-test')
    import cluster_run
    marker = dict(atlas_version=2, iter=10000)
    monkeypatch.setattr(cluster_run, 'read_state', lambda: (None, [], True, marker))
    monkeypatch.setattr(cluster_run, 'read_names', lambda: {'ckpt.pt', 'run.json', 'ckpt_010000.pt'})
    cluster_run.launch(end_iter=15000, dry_run=True)
    output = capsys.readouterr().out
    assert f'--resume {cluster_run.OUT_DIR}/ckpt_010000.pt' in output
    assert '--resume auto' not in output
    monkeypatch.setattr(cluster_run, 'read_names', lambda: {'ckpt.pt', 'run.json'})
    with pytest.raises(ValueError, match='refusing partial ckpt.pt'):
        cluster_run.launch(end_iter=15000, dry_run=True)
    monkeypatch.setattr(cluster_run, 'read_state', lambda: (None, [], True, None))
    with pytest.raises(ValueError, match='without run.json'):
        cluster_run.launch(end_iter=15000, dry_run=True)


def test_eval_requires_completed_numbered_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('DREAMING_RUN_ID', 'eval-test')
    import cluster_run
    monkeypatch.setattr(cluster_run, 'OUT_DIR', str(tmp_path))
    called = []
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: called.append(args))
    monkeypatch.setattr(cluster_run.out, 'commit', lambda: None)
    monkeypatch.setattr(cluster_run.mirror_metrics, 'remote', lambda: None)
    for name in ('ckpt.pt', '../ckpt_010000.pt', 'ckpt_1.pt', 'other.pt'):
        with pytest.raises(ValueError, match='numbered'):
            cluster_run._eval_ckpt(name)
    with pytest.raises(ValueError, match='not complete'):
        cluster_run._eval_ckpt('ckpt_010000.pt')
    (tmp_path / 'run.json').write_text(json.dumps(dict(atlas_version=2, iter=10000)))
    with pytest.raises(ValueError, match='not complete'):
        cluster_run._eval_ckpt('ckpt_015000.pt')
    assert called == []
    (tmp_path / 'ckpt_010000.pt').touch()
    cluster_run._eval_ckpt('ckpt_010000.pt')
    assert called[0][0][called[0][0].index('--checkpoint') + 1] == str(tmp_path / 'ckpt_010000.pt')
    with pytest.raises(TypeError):
        cluster_run.eval()


def test_publish_and_selector(tmp_path, monkeypatch):
    from dreaming import cluster
    from dreaming.web.neural import Checkpoints
    src, root = tmp_path / 'source', tmp_path / 'runs'
    dst = root / 'published-test'
    src.mkdir()
    for name in ('ckpt_005625.pt', 'ckpt_006094.pt'):
        (src / name).write_bytes(name.encode())
    with pytest.raises(ValueError, match='not complete'):
        publish_checkpoints(src, dst, ['ckpt_005625.pt'])
    (src / 'run.json').write_text(json.dumps(dict(atlas_version=2, iter=5625)))
    with pytest.raises(ValueError, match='not complete'):
        publish_checkpoints(src, dst, ['ckpt_006094.pt'])
    (src / 'run.json').write_text(json.dumps(dict(atlas_version=2, iter=6094)))
    replaced = []
    replace = os.replace

    def track(source, target):
        if Path(target).name == 'run.json':
            assert (dst / 'ckpt_005625.pt').exists()
            assert (dst / 'ckpt_006094.pt').exists()
        replaced.append(Path(target).name)
        replace(source, target)

    monkeypatch.setattr(cluster.os, 'replace', track)
    publish_checkpoints(src, dst, ['ckpt_005625.pt', 'ckpt_006094.pt'])
    assert replaced[-1] == 'run.json'
    assert (dst / 'ckpt_006094.pt').read_bytes() == b'ckpt_006094.pt'
    assert json.loads((dst / 'run.json').read_text()) == dict(atlas_version=2, iter=6094)
    publish_checkpoints(src, dst, ['ckpt_005625.pt'])
    assert json.loads((dst / 'run.json').read_text())['iter'] == 6094
    assert replaced.count('ckpt_005625.pt') == 1
    (src / 'ckpt_005625.pt').write_bytes(b'wrong size')
    with pytest.raises(ValueError, match='size mismatch'):
        publish_checkpoints(src, dst, ['ckpt_005625.pt'])
    merge_metrics(src, dst / 'metrics.jsonl')
    catalog = Checkpoints()
    monkeypatch.setattr(catalog, 'load', lambda name: None)
    catalog.configure(str(tmp_path / 'primary'), 'cpu', run_root=str(root))
    assert set(catalog.paths()) == {'published-test/ckpt_005625.pt', 'published-test/ckpt_006094.pt'}
    for name in ('ckpt.pt', '../ckpt_006094.pt', 'ckpt_1.pt', 'ckpt_999999.pt'):
        with pytest.raises(ValueError):
            publish_checkpoints(src, dst, [name])
    for name in ('main', 'atlas_v2_first_5k', 'legacy_v1', 'audit_ab_A'):
        with pytest.raises(ValueError):
            publish_checkpoints(src, root / name, ['ckpt_005625.pt'])
    (dst / 'foreign.txt').touch()
    with pytest.raises(ValueError, match='foreign'):
        publish_checkpoints(src, dst, ['ckpt_005625.pt'])


def test_deadline_kills_descendants(tmp_path):
    pids = tmp_path / 'pids'
    script = ('import os,signal,subprocess,time; '
              'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
              'p=subprocess.Popen(["python","-c","import time; time.sleep(60)"]); '
              f'open({str(pids)!r},"w").write(str(os.getpid())+" "+str(p.pid)); '
              'time.sleep(60)')
    rc, timed_out = run_with_deadline([sys.executable, '-c', script], time.time() + 1, grace=.2)
    assert timed_out and rc == -signal.SIGKILL
    for pid in pids.read_text().split():
        stat = Path(f'/proc/{pid}/stat')
        assert not stat.exists() or stat.read_text().split()[2] == 'Z'


def test_tick_errors_and_normal_exit():
    ticks = []

    def tick():
        ticks.append(1)
        raise RuntimeError('expected tick failure')

    rc, timed_out = run_with_deadline([sys.executable, '-c', 'import time; time.sleep(.4)'],
                                      time.time() + 10, on_tick=tick, tick_every=.1)
    assert (rc, timed_out) == (0, False) and ticks
    with pytest.raises(FileNotFoundError):
        run_with_deadline(['/nonexistent/command'], time.time() + 1)


def test_mirror_and_web_parser(tmp_path):
    from dreaming.web.training import Store
    src, dst = tmp_path / 'source', tmp_path / 'runs' / 'pilot-test' / 'metrics.jsonl'
    merge_metrics(src, dst)
    assert dst.read_text() == ''
    (src / 'eval').mkdir(parents=True)
    train = dict(kind='train', it=5300, lr=.0004, loss=1.)
    evaluation = dict(kind='eval', it=5300, metrics={'doom': {'grid_ce': 1.2,
                      'rollouts': {'2': {'acc_per_step': [.1, .2]}}}})
    (src / 'metrics.jsonl').write_text(json.dumps(train))
    (src / 'eval' / 'metrics.jsonl').write_text(json.dumps(evaluation) + '\n')
    merge_metrics(src, dst)
    assert [json.loads(line) for line in dst.read_text().splitlines()] == [train, evaluation]
    state = Store(str(dst.parent.parent)).parsed(str(dst))
    assert len(state.train) == 1 and state.skipped == 0
    assert state.evals
    (dst.parent / 'ckpt.pt').touch()
    with pytest.raises(ValueError, match='non-metrics'):
        merge_metrics(src, dst)
    with pytest.raises(ValueError, match='destination'):
        merge_metrics(src, dst.with_name('other.jsonl'))


def _ledger(end=None):
    rows = [dict(start=0., deadline=8 * 3600., end=3600.), dict(start=4000., deadline=4000. + 6 * 3600)]
    if end is not None:
        rows[-1]['end'] = end
    return rows


def test_continuation_waits_until_boundary_complete_and_published():
    from dreaming.cluster import continuation_budget
    snap = _ledger()
    assert continuation_budget(_ledger(), snap, 20000, 20000, 20000, 9000., 1e9) is None
    assert continuation_budget(_ledger(9000.), snap, 19000, 19000, 20000, 9000., 1e9) is None
    assert continuation_budget(_ledger(9000.), snap, 20000, 19000, 20000, 9000., 1e9) is None


def test_continuation_budget_sums_ledger_and_respects_stop():
    from dreaming.cluster import continuation_budget
    snap = _ledger()
    used = 3600. + 5000.
    assert continuation_budget(_ledger(9000.), snap, 20000, 20000, 20000, 9100., 1e9) == 8 * 3600 - used - 1
    assert continuation_budget(_ledger(9000.), snap, 20000, 20000, 20000, 9100., 9100. + 7200) == 7200 - 1
    with pytest.raises(ValueError, match='insufficient'):
        continuation_budget(_ledger(9000.), snap, 20000, 20000, 20000, 9100., 9100. + 600)


def test_continuation_refuses_duplicates():
    from dreaming.cluster import continuation_budget
    snap = _ledger()
    extra = _ledger(9000.) + [dict(start=9500., deadline=20000.)]
    for records in (extra, [dict(r, start=r['start'] + 1) for r in _ledger(9000.)]):
        with pytest.raises(ValueError, match='duplicate'):
            continuation_budget(records, snap, 20000, 20000, 20000, 9100., 1e9)
    with pytest.raises(ValueError, match='duplicate'):
        continuation_budget(_ledger(9000.), [], 20000, 20000, 20000, 9100., 1e9)
    with pytest.raises(ValueError, match='advanced'):
        continuation_budget(_ledger(9000.), snap, 21000, 20000, 20000, 9100., 1e9)
    bad = [dict(start=0., deadline=1.)] + _ledger(9000.)[1:]
    with pytest.raises(ValueError, match='never ended'):
        continuation_budget(bad, bad, 20000, 20000, 20000, 9100., 1e9)


def test_continuation_cfg_uses_saved_lock():
    from dreaming.cluster import check_lock, config_row, continuation_cfg
    lock = check_lock(config('overnight-test', extra_shards=400), None)
    cfg = continuation_cfg('overnight-test', lock, 20000, 65000, 1000, 3600., 100.)
    assert cfg['deadline'] == 3700. and cfg['budget_hours'] == 1 and cfg['total_cap_hours'] == 8
    cmd = torchrun_command(cfg, resume=cfg['resume'])
    value = lambda k: cmd[cmd.index(k) + 1]
    assert (value('--extra-shards'), value('--iters'), value('--ckpt-every'), value('--chunk')) == ('400', '65000', '1000', '45000')
    assert value('--resume') == '/out/runs/overnight-test/ckpt_020000.pt' and value('--restart-iter') == '5000'
    assert '--sampler-v2' not in cmd
    assert check_lock(cfg, lock) == lock
    row = config_row(cfg)
    assert row['it'] == 20000 and row['config']['phase_end'] == 65000 and row['config']['checkpoint_every'] == 1000
    with pytest.raises(ValueError):
        continuation_cfg('overnight-test', lock, 20000, 65500, 1000, 3600., 100.)


def test_continuation_treats_null_end_as_active():
    from dreaming.cluster import continuation_budget
    records = _ledger()
    records[-1]['end'] = None
    assert continuation_budget(records, _ledger(), 20000, 20000, 20000, 9000., 1e9) is None


def test_launch_resume_iter_only_pins_latest(monkeypatch, capsys):
    monkeypatch.setenv('DREAMING_RUN_ID', 'resume-test')
    import cluster_run
    marker = dict(atlas_version=2, iter=16000)
    names = {'ckpt.pt', 'run.json', 'ckpt_015000.pt', 'ckpt_016000.pt'}
    monkeypatch.setattr(cluster_run, 'read_state', lambda: (None, [], True, marker))
    monkeypatch.setattr(cluster_run, 'read_names', lambda: names)
    cluster_run.launch(end_iter=65000, ckpt_every=1000, resume_iter=16000, dry_run=True)
    output = capsys.readouterr().out
    assert f'--resume {cluster_run.OUT_DIR}/ckpt_016000.pt' in output and '--chunk 49000' in output
    cluster_run.launch(end_iter=65000, ckpt_every=1000, dry_run=True)
    assert f'--resume {cluster_run.OUT_DIR}/ckpt_016000.pt' in capsys.readouterr().out
    for bad, match in [(17000, 'must be in'), (5000, 'must be in'), (15000, 'overwrite numbered')]:
        with pytest.raises(ValueError, match=match):
            cluster_run.launch(end_iter=65000, ckpt_every=1000, resume_iter=bad, dry_run=True)
    monkeypatch.setattr(cluster_run, 'read_state', lambda: (None, [], False, None))
    with pytest.raises(ValueError, match='requires an existing'):
        cluster_run.launch(end_iter=65000, ckpt_every=1000, resume_iter=15000, dry_run=True)


def test_resume_choice_rejects_bad_types():
    from dreaming.cluster import resume_choice
    assert resume_choice(16000, {'ckpt_016000.pt'}) == (16000, 'ckpt_016000.pt')
    with pytest.raises(ValueError, match='missing'):
        resume_choice(16000, {'ckpt_015000.pt'})
    for bad in (True, 15000.0, -1):
        with pytest.raises(ValueError):
            resume_choice(16000, {'ckpt_015000.pt'}, bad)
