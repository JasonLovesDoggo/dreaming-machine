"""Isolated 3 x 8 B300 launcher. Import and dry-run never hydrate resources."""
import json
import os
from pathlib import Path
import re
import shlex
import time
from datetime import datetime

import modal

from dreaming.cluster import (CalibrationGuard, check_budget, check_lock, config, config_row, resume_choice, continuation_budget,
                              continuation_cfg, lock_config, merge_metrics,
                              publish_checkpoints, publish_set, run_with_deadline,
                              torchrun_command, validate_run_id, write_json)

RUN_ID = os.environ.get('DREAMING_RUN_ID', '')
VOLUME_NAME = validate_run_id(RUN_ID)
app = modal.App('dreaming-cluster')
data = modal.Volume.from_name('dreaming-computer-data')
out = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libibverbs1', 'ibverbs-providers', 'rdma-core', 'librdmacm1')
         .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cu130')
         .run_commands('ln -sf "$(python -c \'import sysconfig; print(sysconfig.get_path("purelib"))\')/nvidia/cu13/lib/libcudart.so.13" /usr/local/lib/libcudart.so && ldconfig')
         .pip_install('numpy', 'pillow')
         .env({'DREAMING_RUN_ID': RUN_ID})
         .add_local_python_source('dreaming')
         .add_local_file('dreaming/apps/dreamer_corpus.json.gz', '/root/dreaming/apps/dreamer_corpus.json.gz'))
OUT_DIR = f'/out/runs/{RUN_ID}'


@app.function(image=image, volumes={'/out': out.read_only(), '/vol': data},
              cpu=1, memory=2048, timeout=1800, max_containers=1)
def mirror_metrics(ckpts: list[str] | None = None):
    out.reload()
    merge_metrics(OUT_DIR, f'/vol/runs/{RUN_ID}/metrics.jsonl')
    publish_checkpoints(OUT_DIR, f'/vol/runs/{RUN_ID}', ckpts or [])
    data.commit()


@app.function(image=image, volumes={'/vol': data.read_only(), '/out': out},
              gpu='B300:8', cpu=64, memory=262144, retries=0, timeout=8 * 3600)
@modal.clustered(size=3, rdma=True)
def train_cluster(cfg: dict):
    start = time.time()
    cluster = modal.Cluster.from_context()
    rank, ips = cluster.container_rank(), cluster.container_ips('ipv4')
    kill_at = min(cfg['deadline'], start + 8 * 3600) - 600
    env = dict(os.environ)
    env.setdefault('NCCL_SOCKET_IFNAME', 'eth0')
    env.setdefault('OMP_NUM_THREADS', '2')
    ledger_path = Path(OUT_DIR) / 'ledger.json'
    records: list[dict] = []
    publish_names = publish_set(cfg, cfg['start_iter'])
    guard = CalibrationGuard(Path(OUT_DIR) / 'metrics.jsonl', cfg, kill_at)
    spawned: set[str] = set()

    def ready():
        marker = Path(OUT_DIR) / 'run.json'
        completed = json.loads(marker.read_text())['iter'] if marker.exists() else -1
        return sorted(name for name in publish_names
                      if int(name[5:-3]) <= completed and (Path(OUT_DIR) / name).exists()
                      and (Path(OUT_DIR) / name).stat().st_mtime >= start)

    if rank == 0:
        lock_config(OUT_DIR, cfg)
        records = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
        check_budget(records, cfg['budget_hours'] * 3600, cfg['total_cap_hours'] * 3600)
        records.append(dict(start=start, deadline=start + cfg['budget_hours'] * 3600))
        write_json(ledger_path, records)
        with (Path(OUT_DIR) / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(config_row(cfg)) + '\n')
        out.commit()

    def tick():
        guard()
        out.commit()
        new = sorted(set(ready()) - spawned)
        mirror_metrics.spawn(new)
        spawned.update(new)

    try:
        rc, timed_out = run_with_deadline(torchrun_command(cfg, rank, ips[0], cfg['resume']), kill_at,
                                          env=env, on_tick=tick if rank == 0 else None)
    finally:
        if rank == 0:
            records[-1]['end'] = time.time()
            write_json(ledger_path, records)
            out.commit()
            mirror_metrics.remote(ready())
    return dict(rank=rank, returncode=rc, timed_out=timed_out, elapsed=time.time() - start)


CONTINUED = 'continuation.json'


@app.function(image=image, volumes={'/vol': data.read_only(), '/out': out},
              cpu=1, memory=2048, retries=0, max_containers=1, timeout=10 * 3600)
def continue_after(after_iter: int, end_iter: int, ckpt_every: int, stop_at: float,
                   wait_seconds: float = 5400, poll: float = 30, handoff: float = 30):
    out_dir, started = Path(OUT_DIR), time.time()

    def read(path, default):
        return json.loads(path.read_text()) if path.exists() else default

    def budget():
        out.reload()
        data.reload()
        if (out_dir / CONTINUED).exists():
            raise ValueError('controller already continued this run')
        records = read(out_dir / 'ledger.json', [])
        out_iter = read(out_dir / 'run.json', {'iter': -1})['iter']
        published = read(Path(f'/vol/runs/{RUN_ID}/run.json'), {'iter': -1})['iter']
        print(json.dumps(dict(event='poll', time=time.time(), out_iter=out_iter, published_iter=published,
                              active_ended=bool(records) and records[-1].get('end') is not None)), flush=True)
        return continuation_budget(records, snapshot, out_iter, published, after_iter, time.time(), stop_at)

    out.reload()
    snapshot = read(out_dir / 'ledger.json', [])
    print(json.dumps(dict(event='controller_start', run_id=RUN_ID, after_iter=after_iter, end_iter=end_iter,
                          ckpt_every=ckpt_every, stop_at=stop_at, wait_seconds=wait_seconds, ledger=snapshot,
                          out_iter=read(out_dir / 'run.json', {'iter': -1})['iter'])), flush=True)
    while budget() is None:
        if time.time() - started > wait_seconds:
            raise TimeoutError(f'iteration {after_iter} not completed and published within {wait_seconds}s')
        time.sleep(poll)
    time.sleep(handoff)
    seconds = budget()
    if seconds is None or not (out_dir / f'ckpt_{after_iter:06d}.pt').exists():
        raise ValueError('boundary state changed during handoff')
    now = time.time()
    cfg = continuation_cfg(RUN_ID, read(out_dir / 'launch.json', None), after_iter, end_iter, ckpt_every, seconds, now)
    write_json(out_dir / CONTINUED, dict(time=now, after_iter=after_iter, cfg=cfg))
    out.commit()
    print(json.dumps(cfg, indent=2), flush=True)
    return train_cluster.remote(cfg)


def _eval_ckpt(ckpt_name: str):
    import subprocess
    if not re.fullmatch(r'ckpt_\d{6}\.pt', ckpt_name):
        raise ValueError('checkpoint must be a numbered ckpt_NNNNNN.pt basename')
    marker = Path(OUT_DIR) / 'run.json'
    if not marker.exists() or int(ckpt_name[5:-3]) > json.loads(marker.read_text())['iter']:
        raise ValueError(f'checkpoint is not complete: {ckpt_name}')
    subprocess.run(['python', '-m', 'dreaming.eval', '--checkpoint', f'{OUT_DIR}/{ckpt_name}',
                    '--data', '/vol/shards', '--holdout-episodes', '2',
                    '--cats-data', '/vol/shards_cats_dense_v1', '--out',
                    f'{OUT_DIR}/eval/{ckpt_name}.json', '--metrics-dir', f'{OUT_DIR}/eval'], check=True)
    out.commit()
    mirror_metrics.remote()


@app.function(image=image, volumes={'/vol': data.read_only(), '/out': out},
              cpu=8, memory=65536, timeout=3600)
def eval_ckpt(ckpt_name: str):
    _eval_ckpt(ckpt_name)


@app.function(image=image, volumes={'/vol': data.read_only(), '/out': out},
              gpu=os.environ.get('DREAMING_EVAL_GPU', 'B300'), cpu=8, memory=65536, timeout=3600)
def eval_ckpt_gpu(ckpt_name: str):
    _eval_ckpt(ckpt_name)


def read_names():
    volume = modal.Volume.from_name(VOLUME_NAME)
    return {e.path.rsplit('/', 1)[-1] for e in volume.listdir(f'/runs/{RUN_ID}')}


def read_state():
    volume = modal.Volume.from_name(VOLUME_NAME)
    try:
        entries = {e.path.rsplit('/', 1)[-1] for e in volume.listdir(f'/runs/{RUN_ID}')}
    except modal.exception.NotFoundError:
        return None, [], False, None

    def read(name, default):
        return json.loads(b''.join(volume.read_file(f'/runs/{RUN_ID}/{name}'))) if name in entries else default
    return read('launch.json', None), read('ledger.json', []), 'ckpt.pt' in entries, read('run.json', None)


@app.local_entrypoint()
def launch(phase: str = 'long', steps: int = 0, end_iter: int = 10000, ckpt_every: int = 5000,
           guard_max_loss: float = 2.0, lr: float = 4e-4, warmup: int = 200,
           iters: int = 65000, budget_hours: float = 8., dry_run: bool = False,
           min_lr_frac: float = 0.1, restart_iter: int = 5000, batch: int = 32,
           unroll: int = 6, data_path: str = '/vol/shards',
           extra_data: str = '/vol/shards_cats_dense_v1', extra_frac: float = 0.15,
           extra_shards: int = 40, holdout_episodes: int = 2, sampler_v2: bool = False,
           source: str = '/vol/runs/atlas_v2_first_5k/ckpt_005000.pt', total_cap_hours: float = 8.,
           resume_iter: int = 0):
    lock, records, resumed, marker = read_state()
    if resumed and marker is None:
        raise ValueError('checkpoint exists without run.json: cannot determine publication start')
    if resume_iter and not resumed:
        raise ValueError('--resume-iter requires an existing resumable run')
    start_iter, latest = resume_choice(marker['iter'], read_names(), resume_iter, restart_iter) if resumed else (restart_iter, '')
    cfg = config(RUN_ID, phase, steps or None, end_iter=end_iter, start_iter=start_iter, ckpt_every=ckpt_every,
                 guard_max_loss=guard_max_loss, lr=lr, warmup=warmup, iters=iters,
                 budget_hours=budget_hours, min_lr_frac=min_lr_frac, restart_iter=restart_iter,
                 batch=batch, unroll=unroll, data=data_path, extra_data=extra_data,
                 extra_frac=extra_frac, extra_shards=extra_shards,
                 holdout_episodes=holdout_episodes, sampler_v2=sampler_v2,
                 source=source, total_cap_hours=total_cap_hours)
    check_lock(cfg, lock)
    check_budget(records, budget_hours * 3600, total_cap_hours * 3600)
    cfg['deadline'] = time.time() + budget_hours * 3600
    cfg['resume'] = f'{OUT_DIR}/{latest}' if resumed else source
    print(json.dumps(cfg, indent=2))
    print(shlex.join(torchrun_command(cfg, resume=cfg['resume'])))
    if not dry_run:
        print(train_cluster.remote(cfg))


@app.local_entrypoint()
def auto_continue(after_iter: int = 20000, end_iter: int = 65000, ckpt_every: int = 1000,
                  stop_at: str = '2026-11-08T14:30:00Z', wait_minutes: float = 90.):
    stop = datetime.fromisoformat(stop_at.replace('Z', '+00:00')).timestamp()
    print(continue_after.remote(after_iter, end_iter, ckpt_every, stop, wait_minutes * 60))


@app.local_entrypoint()
def eval(ckpt_name: str, gpu: bool = False):
    (eval_ckpt_gpu if gpu else eval_ckpt).remote(ckpt_name)


@app.local_entrypoint()
def mirror():
    mirror_metrics.remote()


@app.local_entrypoint()
def publish(ckpts: str):
    names = ckpts.split(',')
    if any(not re.fullmatch(r'ckpt_\d{6}\.pt', name) for name in names):
        raise ValueError('use comma-separated numbered checkpoint basenames')
    mirror_metrics.remote(names)
