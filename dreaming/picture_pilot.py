"""Opt-in complete picture openings; no changes to existing driver modes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .dataset import OBS_KEYS
from .driver import Driver
from .os_sim import KEY_ENTER, MicroOS
from .pictures import CATEGORY_WORDS, IMAGE_EXTS, is_image, parse_name
from .tiles import CANVAS_COLORS, CHARSET, PAL, T

SEED_BASES = {'train': 200_000_000, 'heldout': 210_000_000}
SEED_SPAN = 10_000_000
CORE_NAMES = ('DOG.PNG', 'PEACH.PNG', 'ORANGECAT.PNG', 'ORANGE CAT.PNG')


def episode_seed(split: str, index: int) -> int:
    if split not in SEED_BASES or not 0 <= index < SEED_SPAN:
        raise ValueError('split must be train/heldout and seed index must be in [0, 10000000)')
    return SEED_BASES[split] + index


def filename_for_seed(seed: int) -> str:
    index = seed % SEED_SPAN
    if index % 8 < 4:
        return CORE_NAMES[index % 8]
    categories = [(category, [w for w in words if len(w) <= 13 and all(c in CHARSET for c in w)])
                  for category, words in sorted(CATEGORY_WORDS.items())]
    categories = [(category, words) for category, words in categories if words]
    _, words = categories[(index // 8 * 4 + index % 8 - 4) % len(categories)]
    rng = random.Random(seed)
    return f'{rng.choice(words)}{rng.randrange(100)}{rng.choice(IMAGE_EXTS)}'


def action_stream(driver: Driver, filename: str):
    os_, rng = driver.os, driver.rng
    for _ in range(3):
        yield from driver.open_app('explorer', 'C:/PICS')
        if 'explorer' in os_.z and os_.ex_path == 'C:/PICS':
            break
    else:
        raise RuntimeError(f'driver failed to navigate to Pics: seed {os_.seed}')
    wc, wr, _, _ = driver.win('explorer')
    names = [name for name, _ in os_.fs['C:/PICS']]
    if filename in names:
        yield from driver.click((wc + 3) * T + 4, (wr + 2 + names.index(filename)) * T + 4)
    else:
        yield from driver.click((wc + 9) * T + 4, (wr + 1) * T + 4)
        yield from driver.type_text(filename)
        yield from driver.press_key(KEY_ENTER)
    yield from driver.press_key(KEY_ENTER)
    if os_.pt_file != filename or 'paint' not in os_.z:
        raise RuntimeError(f'driver failed to open {filename}')
    while os_.pt_load < 10:
        driver.x, driver.y = rng.randrange(256), rng.randrange(184)
        yield driver.act()
    yield from driver.idle(rng.randint(10, 20))
    if rng.random() < 0.25:
        yield from driver.ensure_focus('explorer')
        yield from driver.idle(rng.randint(1, 3))
        yield from driver.ensure_focus('paint')
    if rng.random() < 0.3:
        wc, wr, _, _ = driver.win('paint')
        yield from driver.click((wc + rng.randint(3, 8)) * T + 4, (wr + 1) * T + 4)
        if rng.random() < 0.7:
            yield from driver.drag((wc + 3) * T + 2, (wr + 4) * T + 2,
                                   (wc + 8) * T + 2, (wr + 7) * T + 2, speed=12, straight=True)
    yield from driver.idle(rng.randint(2, 5))


def replay_matches(data: dict[str, Any], metadata: dict[str, Any]) -> None:
    if hashlib.sha256(data['action'].tobytes()).hexdigest() != metadata['action_sha256']:
        raise ValueError('action hash mismatch')
    os_ = MicroOS(metadata['seed'])
    observations = os_.observe([])
    for i in range(len(data['action'])):
        for key in OBS_KEYS:
            if not np.array_equal(observations[key], data[key][i]):
                raise ValueError(f'replay mismatch: seed {metadata["seed"]}, tick {i}, {key}')
        if i == metadata['complete_tick']:
            digest = hashlib.sha256(os_.pt_canvas.tobytes()).hexdigest()
            if digest != metadata['completed_canvas_sha256'] or not np.array_equal(os_.pt_canvas, os_.pictures[metadata['filename']]):
                raise ValueError('completed canvas differs from current generator')
        if i + 1 < len(data['action']):
            observations = os_.step(data['action'][i])


def run_episode(seed: int, filename: str | None = None) -> tuple[dict[str, Any], dict[str, Any], Any]:
    filename = (filename or filename_for_seed(seed)).upper()
    if not is_image(filename) or len(filename) > 20 or any(c not in CHARSET for c in filename):
        raise ValueError('filename must be a supported image name of at most 20 driver characters')
    os_ = MicroOS(seed)
    driver = Driver(os_, seed)
    previous = os_.observe([])
    rec = {key: [previous[key]] for key in OBS_KEYS}
    actions: list[Any] = []
    stages: list[int] = []
    counts: Counter[str] = Counter()
    opened = complete = None
    completed_canvas = None
    for action in action_stream(driver, filename):
        before_load = os_.pt_load if 'paint' in os_.z else 0
        obs = os_.step(action)
        driver.observe_events(obs['events'])
        actions.append(action)
        tick = len(actions)
        if 'ex_open_png' in obs['events']:
            if opened is not None:
                raise ValueError('episode opened more than one picture')
            opened = tick
        if opened is not None and os_.pt_load > before_load:
            stages.append(os_.pt_load)
            counts['reveal_updates'] += 1
            counts['newly_revealed_cells'] += (os_.pt_load - before_load) * 16
        if 'paint_loaded' in obs['events']:
            complete = tick
            completed_canvas = os_.pt_canvas.copy()
            if not np.array_equal(completed_canvas, os_.pictures[filename]):
                raise ValueError('loading did not reveal the cached generated picture')
        changed = int(np.count_nonzero(obs['grid'] != previous['grid']))
        counts['changed_cells'] += changed
        counts['idle_cells'] += obs['grid'].size - changed
        counts['idle_ticks'] += int(changed == 0)
        counts['loading_ticks'] += int(opened is not None and os_.pt_load < 10)
        counts['paint_edit_ticks'] += int(any(e in obs['events'] for e in ('paint_color', 'paint_stroke')))
        counts['paint_stroke_ticks'] += int('paint_stroke' in obs['events'])
        counts['other_window_ticks'] += int(opened is not None and os_.focus != 'paint')
        for key in OBS_KEYS:
            rec[key].append(obs[key])
        previous = obs
    if opened is None or complete is None or completed_canvas is None or stages != list(range(1, 11)):
        raise ValueError('episode must contain one complete ten-update picture load')
    actions.append((0, 0, 0, 0))
    data = {key: np.stack(values) for key, values in rec.items()}
    data['action'] = np.array(actions, np.uint8)
    metadata = dict(seed=seed, filename=filename, categories=sorted(set(parse_name(filename)) or {'abstract'}),
                    open_tick=opened, complete_tick=complete, load_stages=stages, ticks=len(actions) - 1,
                    completed_canvas_sha256=hashlib.sha256(completed_canvas.tobytes()).hexdigest(),
                    action_sha256=hashlib.sha256(data['action'].tobytes()).hexdigest(), counts=dict(counts))
    replay_matches(data, metadata)
    return data, metadata, completed_canvas


def source_manifest(source_sha: str | None = None) -> dict[str, Any]:
    root = Path(__file__).parent
    sha = source_sha or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    paths = [root / 'os_sim.py', root / 'driver.py', *sorted((root / 'pictures').rglob('*.py'))]
    return dict(source_sha=sha, source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def contact_sheet(samples: list, path: Path) -> None:
    sheet = Image.new('RGB', (8 * 160, ((len(samples) + 7) // 8) * 112), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (name, seed, canvas) in enumerate(samples):
        x, y = i % 8 * 160, i // 8 * 112
        draw.text((x + 2, y + 2), name, fill='black')
        draw.text((x + 2, y + 14), str(seed), fill='black')
        picture = Image.fromarray(PAL[CANVAS_COLORS][canvas]).resize((128, 80), Image.Resampling.NEAREST)
        sheet.paste(picture, (x + 2, y + 28))
    sheet.save(path)


def generate(out: str, steps: int, split: str, seed_offset: int = 0, workers: int = 1, source_sha: str | None = None):
    if steps <= 0 or not 1 <= workers <= 16:
        raise ValueError('steps must be positive and workers must be in [1, 16]')
    episode_seed(split, seed_offset)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    if list(root.glob('*.npz')):
        raise ValueError('output already contains shards; use a fresh output directory')
    manifest = source_manifest(source_sha)
    counters: dict[str, Counter] = {k: Counter() for k in ('counts', 'picture_opens', 'categories')}
    episodes: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    samples: list[tuple[str, int, Any]] = []
    all_seeds: list[int] = []
    ticks = shard = index = 0

    def flush():
        nonlocal shard
        if not episodes:
            return
        path = root / f'shard_picture_pilot_{split}_{shard:05d}.npz'
        data = {key: np.concatenate([ep[key] for ep in episodes]) for key in episodes[0]}
        data['ep_len'] = np.array([len(ep['action']) for ep in episodes], np.int64)
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **data)
        os.replace(tmp, path)
        path.with_suffix('.npz.json').write_text(json.dumps(dict(mode='picture_pilot', split=split, **manifest,
                                                  seeds=[m['seed'] for m in metadata],
                                                  filenames_opened=[m['filename'] for m in metadata], episodes=metadata), indent=2) + '\n')
        episodes.clear()
        metadata.clear()
        shard += 1

    with ProcessPoolExecutor(max_workers=workers) as pool:
        while ticks < steps:
            seeds = [episode_seed(split, seed_offset + index + i) for i in range(workers)]
            for data, meta, canvas in pool.map(run_episode, seeds):
                episodes.append(data)
                metadata.append(meta)
                all_seeds.append(meta['seed'])
                ticks += meta['ticks']
                counters['counts'].update(meta['counts'])
                counters['picture_opens'][meta['filename']] += 1
                counters['categories'].update(meta['categories'])
                if len(samples) < 48:
                    samples.append((meta['filename'], meta['seed'], canvas))
                if len(episodes) == 25:
                    flush()
                if ticks >= steps:
                    break
            index += workers
    flush()
    contact_sheet(samples, root / 'contact_sheet.png')
    summary = dict(mode='picture_pilot', split=split, **manifest, requested_ticks=steps, ticks=ticks,
                   episodes=len(all_seeds), validated_episodes=len(all_seeds), shards=shard, seeds=all_seeds,
                   **{key: dict(value) for key, value in counters.items()})
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--split', choices=list(SEED_BASES), required=True)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(generate(args.out, args.steps, args.split, args.seed_offset, args.workers), indent=2))
