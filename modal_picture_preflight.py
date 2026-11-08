"""Read-only CPU preflight; only its dedicated smoke volume is writable."""
import json
from pathlib import Path
import time

import modal

app = modal.App('dreaming-picture-preflight')
smoke = modal.Volume.from_name('dreaming-picture-preflight-v1', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cpu')
         .pip_install('numpy', 'pillow')
         .env({'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'})
         .add_local_python_source('dreaming')
         .add_local_file('dreaming/apps/dreamer_corpus.json.gz', '/root/dreaming/apps/dreamer_corpus.json.gz'))


@app.function(image=image, cpu=8, memory=32768, timeout=600, retries=0,
              volumes={'/vol': modal.Volume.from_name('dreaming-computer-data').read_only(),
                       '/src': modal.Volume.from_name('dreaming-run-overnight_20261108_b30024').read_only(),
                       '/pic': modal.Volume.from_name('dreaming-picture-pilot-v1').read_only(), '/smoke': smoke})
def preflight():
    from dreaming.picture_preflight import run_preflight
    report = run_preflight()
    directory = Path('/smoke') / str(time.time_ns())
    directory.mkdir()
    report['report_path'] = str(directory / 'report.json')
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    smoke.commit()
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.local_entrypoint()
def main():
    report = preflight.remote()
    destination = Path.home() / 'picture_preflight_report.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Local report: {destination}; status={report["status"]}')
