"""Modal entrypoints.
  modal run modal_app.py::gen_data --total-steps 20000000
  modal run modal_app.py::train            # 5 nodes x 8 H100, resumes runs/main, trains one 10k chunk
  modal run modal_app.py::gen_data --mode doom --total-steps 4000000 --seed-offset 2000   # specials
  modal run modal_app.py::fetch   (download latest checkpoint + rollout pngs to ./runs/modal)
"""
import modal
import modal.experimental

app = modal.App('dreaming-machine')
vol = modal.Volume.from_name('dreaming-computer-data', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cu130')
         .pip_install('numpy', 'pillow')
         .add_local_python_source('dreaming'))
DATA = '/vol/shards'
RUNS = '/vol/runs'


@app.function(image=image, volumes={'/vol': vol}, cpu=16, memory=16384, timeout=3600)
def gen_shards(seed: int, steps: int, mode: str = 'mixed'):
    from dreaming.dataset import generate
    generate(DATA, steps, ep_len=2000, eps_per_shard=25, workers=16, seed=seed, mode=mode)
    vol.commit()
    return seed


@app.local_entrypoint()
def gen_data(total_steps: int = 20_000_000, containers: int = 20, seed_offset: int = 1000, mode: str = 'mixed'):
    per = total_steps // containers
    seeds = range(seed_offset, seed_offset + containers)
    for s in gen_shards.map(seeds, [per] * containers, [mode] * containers):
        print('done seed', s)


def _train(args: str, gpus: int, dist_args=(), commit=True):
    import subprocess, shlex
    cmd = ['torchrun', f'--nproc_per_node={gpus}', *dist_args, '-m', 'dreaming.train', '--data', DATA] + shlex.split(args)
    print(' '.join(cmd), flush=True)
    smi = subprocess.Popen(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader', '-l', '15'])
    p = subprocess.Popen(cmd)
    import threading, time
    def committer():
        while commit and p.poll() is None:
            time.sleep(300)
            vol.commit()
    threading.Thread(target=committer, daemon=True).start()
    p.wait()
    smi.terminate()
    if commit:
        vol.commit()
    return p.returncode


@app.function(image=image, volumes={'/vol': vol}, gpu='H100:8', memory=262144, cpu=32, timeout=24 * 3600)
def train8(args: str):
    return _train(args, 8)


NODES = 5


@app.function(image=image, volumes={'/vol': vol}, gpu='H100:8', memory=262144, cpu=32, timeout=24 * 3600)
@modal.clustered(size=NODES, rdma=True)
def train_cluster(args: str):
    import os
    import modal.experimental as mx
    info = mx.get_cluster_info()
    os.environ.setdefault('NCCL_SOCKET_IFNAME', 'eth0')
    dist = [f'--nnodes={NODES}', f'--node_rank={info.rank}', f'--master_addr={info.container_ipv4_ips[0]}',
            '--master_port=29500']
    print('cluster rank', info.rank, 'of', NODES, 'master', info.container_ipv4_ips[0], flush=True)
    return _train(args, 8, dist, commit=info.rank == 0)


@app.function(image=image, volumes={'/vol': vol}, gpu='H100:5', memory=196608, cpu=20, timeout=24 * 3600)
def train5(args: str):
    return _train(args, 5)


@app.function(image=image, volumes={'/vol': vol}, gpu='H100', memory=131072, cpu=8, timeout=24 * 3600)
def train1(args: str):
    return _train(args, 1)


@app.local_entrypoint()
def train(gpus: int = 40, iters: int = 150000, chunk: int = 10000, batch: int = 32, unroll: int = 6, run: str = 'main', extra: str = ''):
    args = f'--out {RUNS}/{run} --iters {iters} --chunk {chunk} --batch {batch} --unroll {unroll} --compile --resume auto {extra}'
    if gpus == 8 * NODES:
        print('exit', train_cluster.remote(args))
        return
    fn = {8: train8, 5: train5, 1: train1}[gpus]
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


web_image = (modal.Image.debian_slim(python_version='3.11')
             .pip_install('torch', 'numpy', 'pillow', 'fastapi[standard]')
             .add_local_python_source('dreaming', ignore=['**/__pycache__/**']))


@app.function(image=web_image, gpu=['L4', 'A10G', 'L40S', 'A100', 'H100', 'T4'], volumes={'/vol': vol}, scaledown_window=3600, min_containers=1, timeout=24 * 3600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label='dreaming-computer')
def web():
    from dreaming.web import create_app
    from dreaming.web.telemetry import configure_modal
    configure_modal(vol)
    return create_app(data_dir=DATA, ckpt_dir=f'{RUNS}/main', device='cuda')
