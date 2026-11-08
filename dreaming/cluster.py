"""CPU-only configuration, persistence and process supervision for clustered runs."""
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

SOURCE = '/vol/runs/atlas_v2_first_5k/ckpt_005000.pt'
PROTECTED = {'main', 'atlas_v2_first_5k', 'legacy_v1', 'tput1', 'prof64', 'prof32',
             *(f'audit_ab_{c}' for c in 'ABCDE')}
LOCK_KEYS = ('lr', 'warmup', 'min_lr_frac', 'restart_iter', 'iters', 'batch', 'unroll',
             'data', 'extra_data', 'extra_frac', 'extra_shards', 'holdout_episodes', 'source')


def validate_run_id(run_id):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,47}', run_id) or run_id in PROTECTED:
        raise ValueError('invalid or protected run id')
    name = f'dreaming-run-{run_id}'
    if name in {'dreaming-computer-data', 'dreaming-fallback-v1', 'dreaming-overnight-v1', 'dreaming-audit-ab'}:
        raise ValueError('protected volume')
    return name


def config(run_id, phase='pilot', steps=None, **overrides):
    validate_run_id(run_id)
    presets = {'pilot': (300, 10, 2**63 - 1), 'gate': (700, 10, 1000), 'long': (10000, 25, 10000)}
    n, log, ckpt = presets[phase]
    cfg = dict(run_id=run_id, phase=phase, steps=n if steps is None else steps, log_every=log,
               ckpt_every=ckpt, lr=4e-4, warmup=200, min_lr_frac=0.1, restart_iter=5000,
               iters=65000, batch=32, unroll=6, data='/vol/shards',
               extra_data='/vol/shards_cats_dense_v1', extra_frac=0.15, extra_shards=40,
               holdout_episodes=2, source=SOURCE, budget_hours=8., total_cap_hours=8.)
    cfg.update(overrides)
    if not (0 <= cfg['restart_iter'] < cfg['iters'] and cfg['warmup'] >= 1
            and 0 <= cfg['min_lr_frac'] <= 1 and cfg['steps'] > 0 and cfg['lr'] > 0
            and 0 < cfg['budget_hours'] <= 8 and cfg['total_cap_hours'] > 0):
        raise ValueError('invalid schedule or budget')
    if not all(math.isfinite(float(cfg[k])) for k in ('lr', 'budget_hours', 'total_cap_hours')):
        raise ValueError('non-finite config')
    return cfg


def check_lock(cfg, existing):
    lock = {k: cfg[k] for k in LOCK_KEYS}
    if existing is not None and existing != lock:
        raise ValueError('launch.json schedule mismatch')
    return lock


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def lock_config(out, cfg):
    path = Path(out) / 'launch.json'
    lock = check_lock(cfg, json.loads(path.read_text()) if path.exists() else None)
    if not path.exists():
        write_json(path, lock)


def check_budget(records, requested, cap=8 * 3600):
    used = sum(r.get('end', r['deadline']) - r['start'] for r in records)
    if not 0 < requested <= 8 * 3600 or used + requested > cap:
        raise ValueError('allocation budget exhausted')
    return used


def resume_path(out, source=SOURCE):
    return 'auto' if (Path(out) / 'ckpt.pt').exists() else source


def torchrun_command(cfg, rank=0, master='127.0.0.1', resume=None):
    out = f"/out/runs/{cfg['run_id']}"
    cmd = ['torchrun', '--nnodes=3', '--nproc-per-node=8', f'--node-rank={rank}',
           f'--master-addr={master}', '--master-port=29500', '--max-restarts=0',
           '-m', 'dreaming.warm_restart', '--restart-iter', str(cfg['restart_iter']),
           '--restart-warmup', str(cfg['warmup']), '--min-lr-frac', str(cfg['min_lr_frac']), '--']
    args = {k.replace('_', '-'): cfg[k] for k in ('lr', 'iters', 'batch', 'unroll', 'data',
            'extra_data', 'extra_frac', 'extra_shards', 'holdout_episodes', 'log_every', 'ckpt_every')}
    args.update(out=out, resume=resume if resume is not None else resume_path(out, cfg['source']), chunk=cfg['steps'])
    for key, value in args.items():
        cmd.extend([f'--{key}', str(value)])
    return cmd + ['--compile', '--sync-fix', '--sampler-v2']


def run_with_deadline(cmd, kill_at, env=None, on_tick=None, tick_every=300, grace=60):
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    timed_out = False
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt('SIGTERM')

    def stop():
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            # Wait for the whole group, not only its leader.
            end = time.monotonic() + grace
            while time.monotonic() < end:
                proc.poll()
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    try:
        signal.signal(signal.SIGTERM, terminate)
        next_tick = time.monotonic() + tick_every
        while proc.poll() is None:
            if time.time() >= kill_at:
                timed_out = True
                stop()
                break
            if on_tick and time.monotonic() >= next_tick:
                try:
                    on_tick()
                except Exception:
                    logging.exception('periodic commit/mirror failed')
                next_tick = time.monotonic() + tick_every
            time.sleep(0.1)
    except BaseException:
        stop()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
    return proc.returncode, timed_out


def merge_metrics(src, dst):
    src, dst = Path(src), Path(dst)
    if dst.name != 'metrics.jsonl':
        raise ValueError('destination must be metrics.jsonl')
    if dst.parent.exists() and any(p.name not in {'metrics.jsonl', 'metrics.jsonl.tmp'} for p in dst.parent.iterdir()):
        raise ValueError('refusing to mirror into non-metrics run')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name('metrics.jsonl.tmp')
    with tmp.open('wb') as target:
        for path in (src / 'metrics.jsonl', src / 'eval' / 'metrics.jsonl'):
            if path.exists():
                data = path.read_bytes()
                target.write(data)
                if data and not data.endswith(b'\n'):
                    target.write(b'\n')
    os.replace(tmp, dst)
