"""CPU-only configuration, persistence and process supervision for clustered runs."""
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time

SOURCE = '/vol/runs/atlas_v2_first_5k/ckpt_005000.pt'
PROTECTED = {'main', 'atlas_v2_first_5k', 'legacy_v1', 'tput1', 'prof64', 'prof32',
             *(f'audit_ab_{c}' for c in 'ABCDE')}
LOCK_KEYS = ('lr', 'warmup', 'min_lr_frac', 'restart_iter', 'iters', 'batch', 'unroll',
             'data', 'extra_data', 'extra_frac', 'extra_shards', 'holdout_episodes', 'source', 'sampler_v2')


def validate_run_id(run_id):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,47}', run_id) or run_id in PROTECTED:
        raise ValueError('invalid or protected run id')
    name = f'dreaming-run-{run_id}'
    if name in {'dreaming-computer-data', 'dreaming-fallback-v1', 'dreaming-overnight-v1', 'dreaming-audit-ab'}:
        raise ValueError('protected volume')
    return name


def config(run_id, phase='long', steps=None, end_iter=10000, start_iter=5000, **overrides):
    validate_run_id(run_id)
    if phase != 'long':
        raise ValueError('only the long phase is supported')
    steps = end_iter - start_iter if steps is None else steps
    cfg = dict(run_id=run_id, phase=phase, steps=steps, start_iter=start_iter,
               end_iter=start_iter + steps, log_every=10,
               ckpt_every=5000, sampler_v2=False, guard_max_loss=2.0, lr=4e-4, warmup=200, min_lr_frac=0.1, restart_iter=5000,
               iters=65000, batch=32, unroll=6, data='/vol/shards',
               extra_data='/vol/shards_cats_dense_v1', extra_frac=0.15, extra_shards=40,
               holdout_episodes=2, source=SOURCE, budget_hours=8., total_cap_hours=8.)
    cfg.update(overrides)
    if type(cfg['ckpt_every']) is not int or cfg['ckpt_every'] <= 0:
        raise ValueError('ckpt_every must be a positive int')
    if cfg['end_iter'] % cfg['ckpt_every'] or cfg['end_iter'] > cfg['iters']:
        raise ValueError('end iteration must be a multiple of ckpt_every within the cosine horizon')
    if not (0 <= cfg['restart_iter'] < cfg['iters'] and cfg['warmup'] >= 1
            and 0 <= cfg['min_lr_frac'] <= 1 and cfg['steps'] > 0 and cfg['lr'] > 0
            and 0 < cfg['budget_hours'] <= 8 and cfg['total_cap_hours'] > 0 and cfg['guard_max_loss'] > 0):
        raise ValueError('invalid schedule or budget')
    if not all(math.isfinite(float(cfg[k])) for k in ('lr', 'budget_hours', 'total_cap_hours', 'guard_max_loss')):
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
    return cmd + ['--compile', '--sync-fix'] + (['--sampler-v2'] if cfg['sampler_v2'] else [])


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
                except TrainingGuardError:
                    raise
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
    validate_publish_dir(dst.parent)
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


def validate_publish_dir(path):
    path = Path(path)
    validate_run_id(path.name)
    if path.is_symlink() or (path.exists() and any(
            p.is_symlink() or not p.is_file() or not re.fullmatch(
                r'(metrics\.jsonl|run\.json|ckpt_\d{6}\.pt)(\.tmp)?', p.name)
            for p in path.iterdir())):
        raise ValueError('refusing to mirror into non-metrics run with foreign files')


def publish_checkpoints(src, dst, names):
    src, dst = Path(src), Path(dst)
    validate_publish_dir(dst)
    marker = src / 'run.json'
    completed = json.loads(marker.read_text())['iter'] if marker.exists() else -1
    for name in names:
        if not re.fullmatch(r'ckpt_\d{6}\.pt', name):
            raise ValueError('invalid numbered checkpoint name')
        if not (src / name).is_file() or (src / name).is_symlink():
            raise ValueError(f'checkpoint does not exist: {name}')
        if int(name[5:-3]) > completed:
            raise ValueError(f'checkpoint is not complete: {name}')
        if (dst / name).exists() and (dst / name).stat().st_size != (src / name).stat().st_size:
            raise ValueError(f'published checkpoint size mismatch: {name}')
    if not names:
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        if (dst / name).exists():
            continue
        tmp = dst / (name + '.tmp')
        shutil.copyfile(src / name, tmp)
        os.replace(tmp, dst / name)
    maximum = max(int(p.stem[5:]) for p in dst.glob('ckpt_??????.pt')
                  if re.fullmatch(r'ckpt_\d{6}\.pt', p.name))
    write_json(dst / 'run.json', dict(atlas_version=2, iter=maximum))


def publish_set(cfg, start):
    end = min(start + cfg['steps'], cfg['iters'])
    cadence = cfg['ckpt_every']
    return {f'ckpt_{it:06d}.pt' for it in
            [*range((start // cadence + 1) * cadence, end + 1, cadence), end] if it > start}


class TrainingGuardError(RuntimeError):
    pass


class CalibrationGuard:
    def __init__(self, path, cfg, kill_at):
        self.path, self.cfg, self.kill_at = Path(path), cfg, kill_at
        self.summarized = False

    def __call__(self):
        if not self.path.exists():
            return
        latest = None
        for line in self.path.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):
                continue
            row = json.loads(line)
            if row.get('kind', 'train') != 'train' or row.get('it', 0) <= self.cfg['start_iter']:
                continue
            for key in ('loss', 'grad_norm'):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise TrainingGuardError(f'guard: invalid {key} at iteration {row.get("it")}')
            if (row['it'] > self.cfg['restart_iter'] + self.cfg['warmup']
                    and row['loss'] > self.cfg['guard_max_loss']):
                raise TrainingGuardError(f'guard: loss exceeds {self.cfg["guard_max_loss"]} at {row["it"]}')
            latest = row
        if latest is None or latest['it'] - self.cfg['start_iter'] < 300:
            return
        sps = latest.get('samples_per_sec', 0)
        ups = sps / (self.cfg['batch'] * self.cfg['unroll'] * 24)
        projected = time.time() + (self.cfg['end_iter'] - latest['it']) / ups if ups > 0 else float('inf')
        if not self.summarized:
            print(f'calibration: samples/s={sps:.2f} updates/s={ups:.4f} loss={latest["loss"]:.4f} '
                  f'grad_norm={latest["grad_norm"]:.4f} projected_end={projected:.0f} kill_at={self.kill_at:.0f}', flush=True)
            self.summarized = True
        if projected > self.kill_at:
            logging.warning('projected end %.0f exceeds kill_at %.0f (warning only)', projected, self.kill_at)


def continuation_budget(records, snapshot, out_iter, published_iter, after_iter, now, stop_at,
                        total_cap_hours=8., min_hours=0.5):
    """Seconds for the single continuation, or None while the boundary run is still finishing."""
    if not snapshot or [(r['start'], r['deadline']) for r in records] != [(r['start'], r['deadline']) for r in snapshot]:
        raise ValueError('ledger changed since controller start: refusing duplicate continuation')
    if any(r.get('end') is None for r in records[:-1]):
        raise ValueError('earlier ledger entry never ended')
    if out_iter > after_iter or published_iter > after_iter:
        raise ValueError('run already advanced past the continuation boundary')
    if records[-1].get('end') is None or out_iter < after_iter or published_iter < after_iter:
        return None
    used = sum(r['end'] - r['start'] for r in records)
    budget = min(total_cap_hours * 3600 - used, stop_at - now, 8 * 3600) - 1
    if budget < min_hours * 3600:
        raise ValueError('insufficient budget for continuation')
    return budget


def continuation_cfg(run_id, lock, after_iter, end_iter, ckpt_every, budget, now, total_cap_hours=8.):
    cfg = config(run_id, 'long', end_iter=end_iter, start_iter=after_iter, ckpt_every=ckpt_every,
                 budget_hours=budget / 3600, total_cap_hours=total_cap_hours, **lock)
    check_lock(cfg, lock)
    cfg['deadline'] = now + budget
    cfg['resume'] = f'/out/runs/{run_id}/ckpt_{after_iter:06d}.pt'
    return cfg


def config_row(cfg, world_size=24):
    return dict(kind='config', it=cfg['start_iter'], config=dict(
        world_size=world_size, batch=cfg['batch'], unroll=cfg['unroll'],
        samples_per_update=world_size * cfg['batch'] * cfg['unroll'], log_every=cfg['log_every'],
        checkpoint_every=cfg['ckpt_every'], phase_end=cfg['end_iter'], schedule_end=cfg['iters'],
        restart_iter=cfg['restart_iter'], warmup=cfg['warmup'], peak_lr=cfg['lr'], min_lr_frac=cfg['min_lr_frac']))
