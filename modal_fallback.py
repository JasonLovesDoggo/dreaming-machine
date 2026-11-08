"""Frozen fallback demo: separate app + volume, unaffected by deploys of modal_app.py.
  modal run modal_fallback.py::snapshot     # one-time byte copy runs/main + runs/legacy_v1 -> fallback volume
  modal deploy modal_fallback.py            # serves dreaming-computer-fallback from the read-only fallback volume
"""
import modal

SRC_VOLUME, FALLBACK_VOLUME = 'dreaming-computer-data', 'dreaming-fallback-v1'
RUN_DIRS = ['runs/main', 'runs/legacy_v1']

app = modal.App('dreaming-computer-fallback')
src = modal.Volume.from_name(SRC_VOLUME).read_only()
fallback = modal.Volume.from_name(FALLBACK_VOLUME, create_if_missing=True)

web_image = (modal.Image.debian_slim(python_version='3.11')
             .pip_install('torch', 'numpy', 'pillow', 'fastapi[standard]')
             .env({'DREAMING_TELEMETRY_DIR': '/tmp/telemetry'})
             .add_local_python_source('dreaming', ignore=['**/__pycache__/**']))


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


@app.function(volumes={'/src': src, '/dst': fallback}, timeout=3600)
def copy_runs():
    import os
    import shutil
    out = {}
    for d in RUN_DIRS:
        os.makedirs(f'/dst/{d}', exist_ok=True)
        for name in sorted(os.listdir(f'/src/{d}')):
            s, t = f'/src/{d}/{name}', f'/dst/{d}/{name}'
            shutil.copyfile(s, t)
            a, b = _sha256(s), _sha256(t)
            assert a == b, (s, a, b)
            out[f'{d}/{name}'] = a
    fallback.commit()
    return out


@app.local_entrypoint()
def snapshot():
    for path, digest in copy_runs.remote().items():
        print(digest, path)


@app.function(image=web_image, gpu=['L4', 'A10G', 'L40S', 'A100', 'H100', 'T4'], volumes={'/vol': fallback.read_only()},
              scaledown_window=3600, min_containers=1, timeout=24 * 3600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label='dreaming-computer-fallback')
def web():
    from dreaming.web import create_app
    return create_app(data_dir='/vol/shards', ckpt_dir='/vol/runs/main', device='cuda')
