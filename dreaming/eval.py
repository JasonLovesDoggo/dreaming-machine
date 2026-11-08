"""Deterministic held-out teacher-forcing and autonomous rollout evaluation."""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import OBS_KEYS, generate
from .model import Config, WorldModel, decode
from .tiles import ATLAS, VOCAB, PAL
from .metrics import append_metrics
from .train import Windows, load_training_shards, shard_mode, step_inputs, step_targets, to_torch

MODES = ['mixed', 'doom', 'cards', 'windows', 'calculator', 'terminal', 'dreamer']
HELDOUT_SEED_BASE = 100_000_000


def heldout_paths(directory, exclude_directory='', cats_directory=''):
    excluded = {os.path.basename(p) for p in glob.glob(os.path.join(exclude_directory, '*.npz'))} if exclude_directory else set()
    chosen = {}
    for path in sorted(glob.glob(os.path.join(directory, '*.npz'))):
        mode = shard_mode(path)
        if os.path.basename(path) not in excluded and mode not in chosen:
            chosen[mode] = path
    if cats_directory:
        cats = sorted(glob.glob(os.path.join(cats_directory, '*.npz')))
        if not cats:
            raise ValueError('no cats held-out shard')
        chosen['cats'] = cats[-1]
    if not chosen:
        raise ValueError('no held-out shards')
    return chosen


def mask_grid(out):
    mask = torch.ones(VOCAB, device=out['grid'].device, dtype=torch.bool)
    mask[torch.as_tensor(ATLAS.used, device=mask.device)] = False
    return dict(out, grid=out['grid'].masked_fill(mask, float('-inf')))


def false_change_counts(pred, truth, previous_truth):
    unchanged = truth == previous_truth
    return int(((pred != previous_truth) & unchanged).sum()), int(unchanged.sum())


def changed_transition_counts(pred, truth, previous_truth):
    changed = truth != previous_truth
    sizes = changed.sum(1)
    counts = {}
    for name, frames in [('small', (sizes >= 1) & (sizes <= 16)), ('large', sizes > 16)]:
        selected = changed & frames[:, None]
        counts[name + '_correct'] = int(((pred == truth) & selected).sum())
        counts[name + '_tiles'] = int(selected.sum())
        counts[name + '_frames'] = int(frames.sum())
    return counts


def frame_pair(predicted, truth):
    from .os_sim import obs_to_pixels
    def pixels(obs):
        values = {k: obs[k][0].cpu().numpy() for k in OBS_KEYS}
        values['grid'] = values['grid'].reshape(24, 32)
        return obs_to_pixels(values)
    return PAL[np.concatenate([pixels(predicted), pixels(truth)], axis=1)]


def save_frames(path, frames):
    from PIL import Image, ImageDraw
    height, width = frames[0][1].shape[:2]
    image = Image.new('RGB', (width, (height + 18) * len(frames)))
    draw = ImageDraw.Draw(image)
    for row, (step, pixels) in enumerate(frames):
        top = row * (height + 18)
        draw.text((4, top + 2), f'step {step}: prediction (left) / ground truth (right)', fill='white')
        image.paste(Image.fromarray(pixels), (0, top + 18))
    image.save(path)


@torch.no_grad()
def evaluate_mode(model, data, device, windows=2000, batch=8, starts=20, horizons=(30, 120), seed=7341,
                  dump_frames='', dump_starts=3):
    if windows < 1 or batch < 1 or starts < 1 or not horizons or min(horizons) < 1:
        raise ValueError('evaluation counts and horizons must be positive')
    sampler = Windows(data, 1, np.random.default_rng(seed))
    sampler.p = np.full(len(sampler.valid), 1 / len(sampler.valid))
    totals = dict(ce=0.0, correct=0, changed_correct=0, changed=0, copy=0, tiles=0, false_changes=0)
    totals.update({name + suffix: 0 for name in ('small', 'large') for suffix in ('_correct', '_tiles', '_frames')})
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else torch.autocast('cpu', enabled=False)
    for offset in range(0, windows, batch):
        b = to_torch(sampler.batch(min(batch, windows - offset)), device)
        x, y = step_inputs(b, 0), step_targets(b, 0)
        with amp:
            out, _ = model(x, model.init_memory(len(b['grid'])))
            out = mask_grid(out)
            ce = F.cross_entropy(out['grid'].flatten(0, 1).float(), y['grid'].flatten(), reduction='sum')
        pred = out['grid'].argmax(-1)
        correct = pred == y['grid']
        changed = x['grid'] != y['grid']
        totals['ce'] += float(ce)
        totals['correct'] += int(correct.sum())
        totals['changed_correct'] += int((correct & changed).sum())
        totals['changed'] += int(changed.sum())
        totals['copy'] += int((~changed).sum())
        totals['tiles'] += correct.numel()
        totals['false_changes'] += false_change_counts(pred, y['grid'], x['grid'])[0]
        for key, value in changed_transition_counts(pred, y['grid'], x['grid']).items():
            totals[key] += value
    result = dict(grid_ce=totals['ce'] / totals['tiles'], acc_all=totals['correct'] / totals['tiles'],
                  acc_changed=totals['changed_correct'] / max(1, totals['changed']),
                  changed_tiles=totals['changed'], copy_last_acc_all=totals['copy'] / totals['tiles'],
                  fp_change_rate=totals['false_changes'] / max(1, totals['copy']),
                  unchanged_tiles=totals['copy'], copy_last_fp_change_rate=0.0)
    for name in ('small', 'large'):
        result['acc_changed_' + name] = totals[name + '_correct'] / max(1, totals[name + '_tiles'])
        result['changed_' + name + '_tiles'] = totals[name + '_tiles']
        result['changed_' + name + '_frames'] = totals[name + '_frames']
    result['rollouts'] = {}
    for horizon in horizons:
        rollout_sampler = Windows(data, horizon, np.random.default_rng(seed + horizon))
        rollout_sampler.p = np.full(len(rollout_sampler.valid), 1 / len(rollout_sampler.valid))
        scores, copies = np.zeros(horizon), np.zeros(horizon)
        false_counts, unchanged_counts = np.zeros(horizon), np.zeros(horizon)
        for start in range(starts):
            b = to_torch(rollout_sampler.batch(1), device)
            mem, predicted = model.init_memory(1), None
            first = b['grid'][:, 0]
            frames = []
            dumping = dump_frames and start < dump_starts and horizon == max(horizons)
            if dumping:
                initial = step_inputs(b, 0)
                frames.append((0, frame_pair(initial, initial)))
            for i in range(horizon):
                x = step_inputs(b, i)
                if predicted is not None:
                    x.update({k: predicted[k] for k in OBS_KEYS})
                with amp:
                    out, mem = model(x, mem)
                    predicted = decode(mask_grid(out))
                truth = b['grid'][:, i + 1]
                scores[i] += float((predicted['grid'] == truth).float().mean())
                copies[i] += float((first == truth).float().mean())
                false, unchanged = false_change_counts(predicted['grid'], truth, b['grid'][:, i])
                false_counts[i] += false
                unchanged_counts[i] += unchanged
                if dumping and i + 1 in (10, 50, 100):
                    frames.append((i + 1, frame_pair(predicted, step_targets(b, i))))
            if dumping:
                os.makedirs(dump_frames, exist_ok=True)
                save_frames(os.path.join(dump_frames, f'start_{start:02d}.png'), frames)
        result['rollouts'][str(horizon)] = dict(acc_per_step=(scores / starts).tolist(),
                                                copy_first_acc_per_step=(copies / starts).tolist(),
                                                fp_change_rate_per_step=(false_counts / np.maximum(1, unchanged_counts)).tolist(),
                                                unchanged_tiles_per_step=unchanged_counts.astype(int).tolist(),
                                                copy_last_fp_change_rate_per_step=[0.0] * horizon,
                                                acc_at_steps={str(step): float(scores[step - 1] / starts)
                                                              for step in (10, 50, 100) if step <= horizon},
                                                copy_first_acc_at_steps={str(step): float(copies[step - 1] / starts)
                                                                         for step in (10, 50, 100) if step <= horizon})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--metrics-dir', default='')
    ap.add_argument('--dump-frames', default='', help='PNG strips for first three starts/mode at steps 0/10/50/100')
    ap.add_argument('--exclude-data', default='')
    ap.add_argument('--cats-data', default='')
    ap.add_argument('--holdout-episodes', type=int, default=0)
    ap.add_argument('--generate-heldout', action='store_true', help='generate reserved-base-seed per-app episodes in --data')
    ap.add_argument('--windows', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--starts', type=int, default=20)
    ap.add_argument('--horizons', type=int, nargs='+', default=[30, 120])
    a = ap.parse_args()
    if a.generate_heldout:
        for i, mode in enumerate(MODES):
            generate(a.data, 4000, ep_len=2000, eps_per_shard=2, workers=1, seed=HELDOUT_SEED_BASE + i, mode=mode)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = WorldModel(Config(**ck['cfg'])).to(device).eval()
    model.load_state_dict(ck['model'])
    results = {}
    paths = heldout_paths(a.data, a.exclude_data, a.cats_data)
    for mode, path in paths.items():
        data = load_training_shards([path], holdout_episodes=a.holdout_episodes,
                                    heldout=a.holdout_episodes > 0)
        results[mode] = dict(shard=os.path.basename(path), **evaluate_mode(
            model, data, device, a.windows, a.batch, a.starts, tuple(a.horizons),
            dump_frames=os.path.join(a.dump_frames, mode) if a.dump_frames else ''))
    with open(a.out, 'w') as f:
        json.dump(results, f, indent=2)
    append_metrics(a.metrics_dir or os.path.dirname(os.path.abspath(a.checkpoint)),
                   dict(kind='eval', it=int(ck.get('iter', 0)), time=time.time(),
                        checkpoint=os.path.basename(a.checkpoint), metrics=results))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
