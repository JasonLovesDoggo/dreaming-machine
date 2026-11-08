"""modal run modal_picture_pilot.py — CPU-only, isolated opt-in picture data."""
from collections import Counter
import json
from pathlib import Path
import subprocess

import modal

app = modal.App('dreaming-picture-open-pilot-cpu')
volume = modal.Volume.from_name('dreaming-picture-pilot-v1', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11').pip_install('numpy', 'pillow')
         .env({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
         .add_local_python_source('dreaming')
         .add_local_file('dreaming/apps/dreamer_corpus.json.gz', '/root/dreaming/apps/dreamer_corpus.json.gz'))


@app.function(image=image, volumes={'/pilot': volume}, cpu=16, memory=32768, timeout=3600, max_containers=3)
def partition(split: str, part: int, steps: int, source_sha: str):
    from dreaming.picture_pilot import generate
    path = f'/pilot/{split}/part_{part:02d}'
    result = generate(path, steps, split, seed_offset=part * 100_000, workers=16, source_sha=source_sha)
    volume.commit()
    return dict(path=path, episodes=result['episodes'], ticks=result['ticks'])


@app.function(image=image, volumes={'/pilot': volume}, cpu=16, memory=32768, timeout=600)
def summarize(source_sha: str):
    from PIL import Image, ImageDraw
    volume.reload()
    root = Path('/pilot')
    paths = sorted(root.glob('*/part_*/summary.json'))
    if len(paths) != 3:
        raise ValueError('expected two train partitions and one heldout partition')
    reports = [json.loads(path.read_text()) for path in paths]
    counts = {}
    seeds = {}
    for split in ('train', 'heldout'):
        selected = [r for r in reports if r['split'] == split]
        values: dict[str, Counter[str]] = {key: Counter() for key in ('counts', 'picture_opens', 'categories')}
        seeds[split] = [seed for r in selected for seed in r['seeds']]
        if len(set(seeds[split])) != len(seeds[split]):
            raise ValueError('duplicate seeds within split')
        for report in selected:
            if report['source_sha'] != source_sha or report['source_hashes'] != reports[0]['source_hashes']:
                raise ValueError('source mismatch between partitions')
            if report['validated_episodes'] != report['episodes']:
                raise ValueError('unvalidated episode')
            for key in values:
                values[key].update(report[key])
        counts[split] = dict(episodes=sum(r['episodes'] for r in selected), ticks=sum(r['ticks'] for r in selected),
                             shards=sum(r['shards'] for r in selected), **{k: dict(v) for k, v in values.items()})
    if set(seeds['train']) & set(seeds['heldout']):
        raise ValueError('train and heldout seeds overlap')
    sheets = [Image.open(path.parent / 'contact_sheet.png').convert('RGB') for path in paths]
    sheet = Image.new('RGB', (max(s.width for s in sheets), sum(s.height + 24 for s in sheets)), 'white')
    draw = ImageDraw.Draw(sheet)
    y = 0
    for path, panel in zip(paths, sheets, strict=True):
        draw.text((4, y + 4), str(path.parent.relative_to(root)), fill='black')
        sheet.paste(panel, (0, y + 24))
        y += panel.height + 24
    sheet.save(root / 'contact_sheet.png')
    result = dict(mode='picture_pilot', source_sha=source_sha, source_hashes=reports[0]['source_hashes'],
                  train_heldout_seeds_disjoint=True, validation='Every episode replayed exactly across all OBS_KEYS; completed canvas equals cached current-generator pixels.',
                  splits=counts, partitions=reports, contact_sheet='/contact_sheet.png')
    (root / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    root = Path(__file__).parent
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise ValueError('commit the source before generation so sidecars have exact source provenance')
    source_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    calls = [partition.spawn(split, part, steps, source_sha) for split, part, steps in
             [('train', 0, 75_000), ('train', 1, 75_000), ('heldout', 0, 15_000)]]
    for call in calls:
        print(call.get())
    result = summarize.remote(source_sha)
    print(json.dumps(dict(source_sha=source_sha, counts=result['splits'], contact_sheet=result['contact_sheet']), indent=2))
