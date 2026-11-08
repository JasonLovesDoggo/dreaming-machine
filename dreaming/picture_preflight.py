"""CPU-only inspection of immutable inputs; no volume or launcher operations."""
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

from . import train
from .cluster import (check_budget, check_fork_lock, check_lock, config, fork_source,
                      launch_deadline, torchrun_command)
from .dataset import OBS_KEYS
from .model import Config, WorldModel, STATE_MAX
from .os_sim import MicroOS
from .picture_bank import install_picture_bank, picture_paths

SOURCE_ID = 'overnight_20261108_b30024'
TARGET_ID = 'overnight_20261108_pic24k'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    return dict(sha256=sha256(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def manifest_summaries(value, split=None):
    summaries = []
    if isinstance(value, dict):
        split = value.get('split', split)
        if 'seeds' in value and split in ('train', 'heldout'):
            summaries.append((split, value))
        for key, child in value.items():
            summaries.extend(manifest_summaries(child, key if key in ('train', 'heldout') else split))
    elif isinstance(value, list):
        for child in value:
            summaries.extend(manifest_summaries(child, split))
    return summaries


def shard_stats(data, reference, bounds):
    require(set(data) == set(OBS_KEYS + ['action', 'ep_len']), 'unexpected shard keys')
    lengths = data['ep_len']
    require(np.issubdtype(lengths.dtype, np.integer) and lengths.ndim == 1 and bool((lengths > 1).all()),
            'invalid ep_len')
    n = int(lengths.sum())
    result = {}
    for key in OBS_KEYS + ['action']:
        array = data[key]
        require(len(array) == n, f'{key}: sum(ep_len) mismatch')
        require(array.shape[1:] == reference[key].shape[1:], f'{key}: primary shape mismatch')
        require(array.dtype == reference[key].dtype and np.issubdtype(array.dtype, np.integer),
                f'{key}: primary dtype mismatch')
        require(bool((array >= 0).all()) and bool((array < np.asarray(bounds[key])).all()),
                f'{key}: model vocabulary range violation')
        result[key] = dict(dtype=str(array.dtype), shape=list(array.shape), min=int(array.min()), max=int(array.max()))
    return dict(arrays=result, frames=n, ticks=n - len(lengths), episodes=len(lengths))


def replay_matches(data, metadata):
    """PR #32 picture_pilot.replay_matches, bundled without its generator/CLI."""
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


def check_sidecar(path, data, metadata, heldout_seeds):
    require(metadata['split'] == 'train', 'non-train sidecar')
    seeds = metadata['seeds']
    require(len(seeds) == len(set(seeds)) == len(data['ep_len']), 'sidecar seed count mismatch')
    require(all(200000000 <= seed < 210000000 for seed in seeds), 'train seed out of range')
    require(not set(seeds) & heldout_seeds, 'train/heldout seed overlap')
    if 'sha256' in metadata:
        require(metadata['sha256'] == sha256(path), 'shard sha256 mismatch')
    if 'shard_sha256' in metadata:
        require(metadata['shard_sha256'] == sha256(path), 'shard sha256 mismatch')
    require(len(metadata['episodes']) == len(seeds), 'episode metadata count mismatch')
    offset = 0
    for seed, length, episode in zip(seeds, data['ep_len'], metadata['episodes']):
        require(seed == episode['seed'] and episode['ticks'] == int(length) - 1, 'episode count/seed mismatch')
        action_hash = hashlib.sha256(data['action'][offset:offset + length].tobytes()).hexdigest()
        require(action_hash == episode['action_sha256'], 'episode action sha256 mismatch')
        offset += int(length)
    return seeds


def model_bounds(model):
    return dict(grid=model.tile.num_embeddings, cursor=[model.cx.num_embeddings, model.cy.num_embeddings],
                action=[model.ax.num_embeddings, model.ay.num_embeddings, model.abtn.num_embeddings, model.akey.num_embeddings],
                doom_cols=[model.dtype_.num_embeddings, model.dside.num_embeddings, model.dh.num_embeddings, model.du.num_embeddings],
                doom_spr=[model.skind.num_embeddings, model.sx.num_embeddings, model.ssize.num_embeddings],
                doom_meta=[model.mgun.num_embeddings, model.mvc.num_embeddings, model.mvr.num_embeddings], state=STATE_MAX)


def inspect_pictures(root, reference, model, report):
    manifest = json.loads((root / 'manifest.json').read_text())
    report.update(tree=sorted(str(path.relative_to(root)) for path in root.rglob('*')), manifest=manifest)
    summaries = manifest_summaries(manifest)
    heldout = {seed for split, summary in summaries if split == 'heldout' for seed in summary['seeds']}
    expected = {seed for split, summary in summaries if split == 'train' for seed in summary['seeds']}
    require(bool(heldout) and bool(expected), 'manifest lacks identifiable train/heldout seed summaries')
    paths = picture_paths(f'{root}/train/part_00,{root}/train/part_01')
    report.update(shards=[], replay=dict(origin='PR #32 replay_matches (bundled)', episodes=0), bounds=model_bounds(model))
    seeds_seen = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        stats = shard_stats(data, reference, report['bounds'])
        stats['path'] = path
        report['shards'].append(stats)
        sidecar = json.loads(Path(path + '.json').read_text())
        seeds = check_sidecar(path, data, sidecar, heldout)
        stats['seeds'] = seeds
        stats['counts'] = dict(sum((Counter(episode['counts']) for episode in sidecar['episodes']), Counter()))
        stats['picture_opens'] = dict(Counter(episode['filename'] for episode in sidecar['episodes']))
        stats['categories'] = dict(Counter(category for episode in sidecar['episodes'] for category in episode['categories']))
        stats['action_hashes_verified'] = len(seeds)
        require(not seeds_seen.intersection(seeds), 'duplicate train seeds across shards')
        seeds_seen.update(seeds)
        for _, summary in summaries:
            if set(seeds).issubset(set(summary['seeds'])):
                for key in ('source_sha', 'source_hashes'):
                    if key in summary:
                        require(sidecar[key] == summary[key], f'{key}: sidecar/manifest mismatch')
        offset = 0
        for length, episode in zip(data['ep_len'], sidecar['episodes']):
            if report['replay']['episodes'] < 20:
                replay_matches({key: data[key][offset:offset + length] for key in OBS_KEYS + ['action']}, episode)
                report['replay']['episodes'] += 1
            offset += int(length)
    require(seeds_seen == expected, 'manifest/train shard seed set mismatch')
    for split, summary in summaries:
        if split == 'train':
            matching = [s for s in report['shards'] if set(s['seeds']).issubset(set(summary['seeds']))]
            for key in ('ticks', 'episodes', 'shards'):
                if key in summary:
                    actual = len(matching) if key == 'shards' else sum(s[key] for s in matching)
                    require(actual == summary[key], f'manifest {key} mismatch')
            for key in ('counts', 'picture_opens', 'categories'):
                if key in summary:
                    actual_counts = dict(sum((Counter(s[key]) for s in matching), Counter()))
                    require(actual_counts == summary[key], f'manifest {key} mismatch')
    report.update(total_ticks=sum(s['ticks'] for s in report['shards']), total_frames=sum(s['frames'] for s in report['shards']),
                  total_episodes=len(seeds_seen), no_heldout_loaded=True)
    require(abs(report['total_ticks'] - 150079) < 5000, 'unexpected total train ticks')


def loader_args(lock):
    keys = ('batch', 'unroll', 'lr', 'warmup', 'iters', 'data', 'extra_data', 'extra_shards', 'extra_frac', 'holdout_episodes')
    argv = ['preflight', '--max-steps-data', '20000']
    for key in keys:
        argv.extend([f'--{key.replace("_", "-")}', str(lock[key])])
    if lock.get('sampler_v2'):
        argv.append('--sampler-v2')
    original = sys.argv
    try:
        sys.argv = argv
        return train.parse_args()
    finally:
        sys.argv = original


def loader_probe(args, picture_dirs):
    original = train.load_data
    try:
        baseline = original(args, 0, 24).batch(32)
        install_picture_bank(train, picture_dirs, .15, 0)
        windows = train.load_data(args, 0, 24)
    finally:
        train.load_data = original
    counts = dict(primary=0, cats=0, pictures=0)
    for name, bank in zip(counts, (windows.mixed.primary, windows.mixed.extra, windows.pictures)):
        batch = bank.batch

        def counted(size, name=name, batch=batch):
            counts[name] += size
            return batch(size)
        bank.batch = counted
    result = windows.batch(32)
    require(counts == dict(primary=22, cats=5, pictures=5), 'incorrect three-bank counts')
    require(set(result) == set(baseline), 'two/three bank key mismatch')
    for key in result:
        require(result[key].shape == baseline[key].shape and result[key].dtype == baseline[key].dtype,
                f'{key}: two/three bank shape/dtype mismatch')
        require(result[key].shape[:2] == (32, 7), f'{key}: expected [32, 7, ...]')
    small = {key: array[[0, 22, 27], :2].copy() for key, array in result.items()}
    return small, dict(args=vars(args), counts=counts, arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                                                            for key, value in result.items()})


def optimizer_report(state):
    steps = []
    tensors = 0
    for item in state['state'].values():
        for value in item.values():
            if torch.is_tensor(value):
                require(bool(torch.isfinite(value).all()), 'nonfinite optimizer tensor')
                tensors += 1
        steps.append(float(item['step']))
    require(len(steps) > 0 and len(set(steps)) == 1, 'inconsistent optimizer steps')
    return dict(param_states=len(steps), tensor_count=tensors, step=steps[0], steps_consistent=True)


def model_step(model, checkpoint, args, batch):
    args.unroll = 1
    optimizer = train.build_optimizer(model, args, 'cpu')
    optimizer.load_state_dict(checkpoint['opt'])
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = train.unroll_loss(model, model, train.to_torch(batch, 'cpu'), args, 0., 0.,
                             torch.Generator().manual_seed(123), 'cpu', {})
    require(bool(torch.isfinite(loss)), 'nonfinite loss')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    require(bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients), 'nonfinite/missing gradients')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    require(bool(torch.isfinite(norm)), 'nonfinite grad norm')
    optimizer.step()
    changed = sum(not torch.equal(before[key], value) for key, value in model.named_parameters())
    require(changed > 0 and all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'invalid updated parameters')
    return dict(loss=float(loss.detach()), grad_norm=float(norm), changed_parameters=changed,
                gradient_tensors=len(gradients), finite=True, samples=3, unroll=1,
                path='train.unroll_loss + backward + clip_grad_norm_(1.0) + train.build_optimizer + opt.step',
                scheduled_sampling=0, corruption=0, optimizer_after=optimizer_report(optimizer.state_dict()))


def sandbox(lock, marker, ledger, names, now):
    records = [dict(record) for record in ledger]
    prospective_stop = now + 900
    records[-1]['end'] = prospective_stop
    source, inherited = fork_source(SOURCE_ID, TARGET_ID, marker['iter'], marker, names, records)
    settings = {**lock, 'source': source, 'picture_data': '/pic/train/part_00,/pic/train/part_01', 'picture_frac': .15}
    end = max(65000, (marker['iter'] // 1000 + 1) * 1000)
    cfg = config(TARGET_ID, start_iter=marker['iter'], end_iter=end, ckpt_every=1000, **settings)
    check_fork_lock(cfg, lock)
    check_lock(cfg, check_lock(cfg, None))
    used = sum(r['end'] - r['start'] for r in records)
    remaining = 8 * 3600 - used
    cases = []
    for stop in ('2026-11-08T14:29:00Z', '2026-11-08T14:39:18Z'):
        row: dict[str, Any] = dict(stop_at=stop, stop_minus_cleanup=datetime.fromtimestamp(
            datetime.fromisoformat(stop.replace('Z', '+00:00')).timestamp() - 600, timezone.utc).isoformat())
        try:
            budget = min(8 * 3600, remaining)
            check_budget(inherited, budget)
            deadline = launch_deadline(prospective_stop, budget / 3600, stop)
            kill_at = min(deadline, prospective_stop + 8 * 3600) - 600
            row.update(budget_hours=budget / 3600, deadline=deadline, kill_at=kill_at,
                       kill_at_utc=datetime.fromtimestamp(kill_at, timezone.utc).isoformat(),
                       budget_clamps_earlier=deadline < datetime.fromisoformat(stop.replace('Z', '+00:00')).timestamp())
        except ValueError as error:
            row['refused'] = str(error)
        cases.append(row)
    return dict(prospective_stop=prospective_stop, source_records_changed_in_memory_only=True,
                used_h=used / 3600, remaining_cap_h=remaining / 3600, cases=cases,
                original_deadline=ledger[-1].get('deadline'),
                original_allocation_hours=(ledger[-1]['deadline'] - ledger[-1]['start']) / 3600 if 'deadline' in ledger[-1] else None,
                torchrun_command=shlex.join(torchrun_command(cfg, resume=source)))


def run_preflight(source='/src/runs/overnight_20261108_b30024', pictures='/pic', primary='/vol/shards'):
    start = time.time()
    report: dict[str, Any] = dict(status='running', container_start=start, device='cpu')
    path = None
    torch.set_num_threads(8)
    try:
        root = Path(source)
        lock, marker, ledger = (json.loads((root / name).read_text()) for name in ('launch.json', 'run.json', 'ledger.json'))
        report['source'] = dict(launch=lock, run=marker, ledger=ledger, last_record_active=ledger[-1].get('end') is None)
        path = root / f'ckpt_{marker["iter"]:06d}.pt'
        report['checkpoint'] = dict(path=str(path), before=fingerprint(path))
        ck = torch.load(path, map_location='cpu')
        require(ck['iter'] == marker['iter'] == int(path.stem.split('_')[1]), 'checkpoint iteration mismatch')
        model = WorldModel(Config(**ck['cfg']))
        model.load_state_dict(ck['model'], strict=True)
        require(all(bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()), 'nonfinite model tensors')
        report['checkpoint'].update(iter=ck['iter'], cfg=ck['cfg'], strict_model_load=True, model_finite=True,
                                    optimizer=optimizer_report(ck['opt']))
        require(report['checkpoint']['optimizer']['param_states'] == 187, 'expected 187 optimizer param states')
        primary_path = next(iter(sorted(Path(primary).glob('*.npz'))))
        with np.load(primary_path, allow_pickle=False) as archive:
            reference = {key: archive[key] for key in archive.files}
        report['primary'] = dict(path=str(primary_path), statistics=shard_stats(reference, reference, model_bounds(model)))
        report['pictures'] = {}
        inspect_pictures(Path(pictures), reference, model, report['pictures'])
        args = loader_args(lock)
        small, report['loader'] = loader_probe(args, f'{pictures}/train/part_00,{pictures}/train/part_01')
        report['model_step'] = model_step(model, ck, args, small)
        report['sandbox'] = sandbox(lock, marker, ledger, {p.name for p in root.iterdir()}, time.time())
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        if path is not None and 'checkpoint' in report:
            report['checkpoint']['after'] = fingerprint(path)
            report['checkpoint']['unchanged'] = report['checkpoint']['before'] == report['checkpoint']['after']
            if not report['checkpoint']['unchanged']:
                report.update(status='failed', error='source checkpoint fingerprint changed')
        report['elapsed_seconds'] = time.time() - start
    return report
