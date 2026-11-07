"""Modal entrypoints.
  modal run modal_app.py::gen_data --total-steps 20000000
  modal run modal_app.py::train --gpus 8 --iters 30000
  modal run modal_app.py::fetch   (download latest checkpoint + rollout pngs to ./runs/modal)
"""
import modal

app = modal.App('dreaming-machine')
vol = modal.Volume.from_name('dreaming-computer-data', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.5.1', 'numpy', 'pillow')
         .add_local_python_source('dreaming'))
DATA = '/vol/shards'
RUNS = '/vol/runs'


@app.function(image=image, volumes={'/vol': vol}, cpu=16, memory=16384, timeout=3600)
def gen_shards(seed: int, steps: int):
    from dreaming.dataset import generate
    generate(DATA, steps, ep_len=2000, eps_per_shard=25, workers=16, seed=seed)
    vol.commit()
    return seed


@app.local_entrypoint()
def gen_data(total_steps: int = 20_000_000, containers: int = 20):
    per = total_steps // containers
    for s in gen_shards.map(range(containers), [per] * containers):
        print('done seed', s)


def _train(args: str, gpus: int):
    import subprocess, shlex
    cmd = ['torchrun', f'--nproc_per_node={gpus}', '-m', 'dreaming.train', '--data', DATA] + shlex.split(args)
    print(' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd)
    import threading, time
    def committer():
        while p.poll() is None:
            time.sleep(300)
            vol.commit()
    threading.Thread(target=committer, daemon=True).start()
    p.wait()
    vol.commit()
    return p.returncode


@app.function(image=image, volumes={'/vol': vol}, gpu='H100:8', memory=262144, cpu=32, timeout=24 * 3600)
def train8(args: str):
    return _train(args, 8)


@app.function(image=image, volumes={'/vol': vol}, gpu='H100', memory=131072, cpu=8, timeout=24 * 3600)
def train1(args: str):
    return _train(args, 1)


@app.local_entrypoint()
def train(gpus: int = 8, iters: int = 30000, batch: int = 16, unroll: int = 6, run: str = 'main', extra: str = ''):
    args = f'--out {RUNS}/{run} --iters {iters} --batch {batch} --unroll {unroll} --compile {extra}'
    fn = train8 if gpus == 8 else train1
    print('exit', fn.remote(args))


@app.local_entrypoint()
def fetch(run: str = 'main'):
    import os
    os.makedirs(f'runs/modal/{run}', exist_ok=True)
    for e in vol.listdir(f'runs/{run}'):
        name = e.path.split('/')[-1]
        with open(f'runs/modal/{run}/{name}', 'wb') as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        print('fetched', name)
