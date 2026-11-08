"""Train the world model. Single GPU: python -m dreaming.train --data data/local
Multi GPU: torchrun --nproc_per_node 8 -m dreaming.train --data /data/shards"""
import argparse
import glob
import json
import math
import os
import threading
import queue
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .dataset import OBS_KEYS
from .model import WorldModel, Config, compute_loss, decode
from .tiles import VOCAB, PAL, ATLAS
from .metrics import append_metrics


class Windows:
    """Random length-(L+1) windows that stay inside one episode."""

    def __init__(self, data, L, rng, sampler_v2=False, mode_mix=None):
        self.d, self.L, self.rng = data, L, rng
        ends = np.cumsum(data['ep_len'])
        starts = ends - data['ep_len']
        self.valid = np.concatenate([np.arange(s, e - L - 1) for s, e in zip(starts, ends)])
        if not len(self.valid):
            raise ValueError('no episodes long enough for the requested unroll')
        self.ep_start = np.repeat(starts, data['ep_len'])
        g = data['grid'].reshape(len(data['grid']), -1)
        changed = np.zeros(len(g), np.float64)
        changed[1:] = (g[1:] != g[:-1]).sum(1)
        if sampler_v2:
            for key, scale in [('doom_cols', 4), ('doom_spr', 4), ('doom_meta', 1), ('cursor', 1), ('state', 1)]:
                values = data[key].reshape(len(g), -1)
                changed[1:] += (values[1:] != values[:-1]).sum(1) / scale
        cs = np.concatenate([[0], np.cumsum(changed)])
        act = cs[self.valid + L + 1] - cs[self.valid + 1]
        w = 1 + np.minimum(act, 10) if sampler_v2 else 0.25 + np.minimum(act, 50)
        if sampler_v2:
            modes = data['mode']
            available = set(modes[self.valid])
            if mode_mix is not None:
                if set(mode_mix) - available:
                    raise ValueError('mode mix names modes with no eligible windows')
                if any(not np.isfinite(v) or v < 0 for v in mode_mix.values()) or not sum(mode_mix.values()):
                    raise ValueError('mode mix must contain finite nonnegative weights with positive total')
            for mode in available:
                mask = modes[self.valid] == mode
                mass = (modes == mode).mean() if mode_mix is None else mode_mix.get(mode, 0)
                w[mask] *= mass / w[mask].sum()
        self.p = w / w.sum()

    def batch(self, B):
        s = self.rng.choice(self.valid, B, p=self.p)
        idx = s[:, None] + np.arange(self.L + 1)[None]
        out = {k: self.d[k][idx] for k in OBS_KEYS}
        out['action'] = self.d['action'][idx]
        prev = np.where(idx - 1 >= self.ep_start[idx], self.d['action'][np.maximum(idx - 1, 0), 2], 0)
        out['prev_btn'] = prev.astype(np.uint8)
        return out


def shard_mode(path):
    if os.path.exists(path + '.json'):
        with open(path + '.json') as f:
            mode = json.load(f).get('mode')
        if mode:
            return mode
    name = os.path.basename(path)
    return next((m for m in ['doom', 'cards', 'windows', 'calculator', 'terminal', 'dreamer', 'cats'] if m in name), 'mixed')


def partition_shards(paths, rank, world):
    if world < 1 or not 0 <= rank < world:
        raise ValueError('shard partition requires world >= 1 and 0 <= rank < world')
    if not paths:
        raise ValueError('shard partition would give every rank zero shards')
    if len(paths) < world:
        return paths
    modes = {}
    for path in paths:
        modes.setdefault(shard_mode(path), []).append(path)
    partitions = [[] for _ in range(world)]
    counter = 0
    for mode in sorted(modes):
        for path in sorted(modes[mode]):
            partitions[counter % world].append(path)
            counter += 1
    if any(not part for part in partitions):
        raise ValueError('shard partition would give a rank zero shards')
    return partitions[rank]


class MixedWindows:
    def __init__(self, primary, extra, frac):
        self.primary, self.extra, self.frac = primary, extra, frac

    def batch(self, B):
        n = round(B * self.frac)
        if not n:
            return self.primary.batch(B)
        if n == B:
            return self.extra.batch(B)
        a, b = self.primary.batch(B - n), self.extra.batch(n)
        return {k: np.concatenate([a[k], b[k]]) for k in a}


def to_torch(b, device):
    t = {}
    for k, v in b.items():
        x = torch.from_numpy(v.astype(np.int64))
        if k == 'grid':
            x = x.flatten(-2)
        t[k] = x.to(device, non_blocking=True)
    return t


def step_inputs(b, i, grid_override=None):
    x = {k: b[k][:, i] for k in OBS_KEYS + ['action', 'prev_btn']}
    if grid_override is not None:
        x['grid'] = grid_override
    return x


def step_targets(b, i):
    return {k: b[k][:, i + 1] for k in OBS_KEYS}


def _pixels(obs):
    from .os_sim import obs_to_pixels
    o = {k: v[0].cpu().numpy() for k, v in obs.items()}
    o['grid'] = o['grid'].reshape(24, 32)
    return obs_to_pixels(o)


def save_rollout_png(path, model, b, mem0, L):
    """Rows of [ground truth | teacher-forced | free-running] for the first sample of the batch."""
    from PIL import Image
    rows = []
    with torch.no_grad():
        mem_tf = mem_free = mem0[:1]
        free = None
        for i in range(L):
            x = {k: v[:1] for k, v in step_inputs(b, i).items()}
            out_tf, mem_tf = model(x, mem_tf)
            x_free = dict(x, **free) if free is not None else x
            out_free, mem_free = model(x_free, mem_free)
            free = {k: v for k, v in decode(out_free).items() if k in OBS_KEYS}
            truth = {k: v[:1] for k, v in step_targets(b, i).items()}
            rows.append(np.concatenate([_pixels(truth), _pixels(decode(out_tf)), _pixels(free)], 1))
    Image.fromarray(PAL[np.concatenate(rows[::max(1, L // 6)], 0)]).save(path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/local')
    ap.add_argument('--out', default='runs/local')
    ap.add_argument('--max-steps-data', type=int, default=0)
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--chunk', type=int, default=0, help='stop after this many iters (resume later); LR schedule still uses --iters')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--unroll', type=int, default=6)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--d', type=int, default=512)
    ap.add_argument('--layers', type=int, default=12)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--mem', type=int, default=32)
    ap.add_argument('--ss-max', type=float, default=0.5, help='max scheduled-sampling prob')
    ap.add_argument('--corrupt-max', type=float, default=0.03)
    ap.add_argument('--log-every', type=int, default=25)
    ap.add_argument('--ckpt-every', type=int, default=1000)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--resume', default='')
    ap.add_argument('--sync-fix', action='store_true', help='accumulate detached loss metrics on device until logging')
    ap.add_argument('--sampler-v2', action='store_true', help='modality-aware sampling with idle floor and mode balancing')
    ap.add_argument('--mode-mix', type=json.loads, default=None, help='JSON object of mode sampling masses; default: frame shares')
    ap.add_argument('--extra-data', default='')
    ap.add_argument('--extra-shards', type=int, default=40, help='shard cap (1..40); final shard is always reserved for evaluation')
    ap.add_argument('--extra-frac', type=float, default=0.15, help='extra-source batch fraction; future cats target: 0.15..0.20')
    ap.add_argument('--holdout-episodes', type=int, default=0, help='reserve final N episodes of each shard for evaluation')
    return ap.parse_args()


def setup_distributed():
    """-> (rank, world, ddp). torchrun sets WORLD_SIZE / LOCAL_RANK."""
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group('nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        return dist.get_rank(), dist.get_world_size(), True
    return 0, 1, False


def load_data(a, rank, world):
    paths = sorted(glob.glob(os.path.join(a.data, '*.npz')))
    paths = partition_shards(paths, rank, world)
    t0 = time.time()
    def windows(paths, seed, mode_mix=None):
        data = load_training_shards(paths, a.max_steps_data or None, a.holdout_episodes)
        return Windows(data, a.unroll, np.random.default_rng(seed), a.sampler_v2, mode_mix)
    if a.mode_mix is not None and (not a.sampler_v2 or not isinstance(a.mode_mix, dict)):
        raise ValueError('--mode-mix requires --sampler-v2 and a JSON object')
    win = windows(paths, 1234 + rank, a.mode_mix)
    if rank == 0:
        print(f'loaded {len(win.d["action"])} steps/rank from {len(paths)} shards in {time.time() - t0:.0f}s', flush=True)
    if a.extra_data:
        if not 0 <= a.extra_frac <= 1 or not 1 <= a.extra_shards <= 40:
            raise ValueError('extra fraction must be in [0,1] and shard cap in [1,40]')
        extra = sorted(glob.glob(os.path.join(a.extra_data, '*.npz')))[:-1][:a.extra_shards]
        extra = partition_shards(extra, rank, world)
        win = MixedWindows(win, windows(extra, 5678 + rank), a.extra_frac)
    return win


def load_training_shards(paths, max_steps=None, holdout_episodes=0, heldout=False):
    if holdout_episodes < 0:
        raise ValueError('holdout episode count must be nonnegative')
    parts, total = [], 0
    for path in paths:
        with np.load(path) as z:
            lengths = z['ep_len']
            if holdout_episodes >= len(lengths):
                raise ValueError('holdout must leave at least one training episode per shard')
            cut = len(lengths) - holdout_episodes
            boundary = int(lengths[:cut].sum())
            sl = slice(boundary, None) if heldout else slice(None, boundary)
            part = {k: z[k][sl] for k in OBS_KEYS + ['action']}
            part['ep_len'] = lengths[cut:] if heldout else lengths[:cut]
            part['mode'] = np.full(len(part['action']), shard_mode(path))
        parts.append(part)
        total += len(part['action'])
        if max_steps and total >= max_steps:
            break
    if not parts or not total:
        raise ValueError('no eligible data shards or episodes')
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load_checkpoint(a):
    """Resolve --resume (auto = <out>/ckpt.pt if present) -> checkpoint dict or None."""
    path = a.resume
    if path == 'auto':
        path = os.path.join(a.out, 'ckpt.pt')
        if not os.path.exists(path):
            return None
    return torch.load(path, map_location='cpu') if path else None


def build_optimizer(model, a, device):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{'params': decay, 'weight_decay': 0.05}, {'params': no_decay, 'weight_decay': 0}],
                             lr=a.lr, betas=(0.9, 0.95), fused=device == 'cuda')


def lr_at(it, a):
    """Linear warmup, then cosine decay to 10% over --iters."""
    return a.lr * min(1, (it + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / a.iters)))


def start_loader(win, batch, n_threads=2):
    q = queue.Queue(maxsize=4)

    def fill():
        while True:
            q.put(win.batch(batch))
    for _ in range(n_threads):
        threading.Thread(target=fill, daemon=True).start()
    return q


def unroll_loss(fwd, model, b, a, p_ss, p_cor, gen, device, agg):
    """Run the recurrent unroll with scheduled sampling + input corruption on the grid; returns mean loss."""
    B = b['grid'].shape[0]
    used = torch.as_tensor(ATLAS.used, device=device)
    mem, total, prev_pred = model.init_memory(B), 0, None
    for i in range(a.unroll):
        g_in = b['grid'][:, i]
        if prev_pred is not None and p_ss > 0:
            use_pred = torch.rand(B, 1, device=device, generator=gen) < p_ss
            g_in = torch.where(use_pred, prev_pred, g_in)
        if p_cor > 0:
            rate = p_cor * torch.rand(B, 1, device=device, generator=gen)
            noise = torch.rand(g_in.shape, device=device, generator=gen) < rate
            g_in = torch.where(noise, used[torch.randint(0, len(used), g_in.shape, device=device, generator=gen)], g_in)
        out, mem = fwd(step_inputs(b, i, g_in), mem)
        loss, parts = compute_loss(out, step_targets(b, i), b['grid'][:, i])
        total = total + loss / a.unroll
        prev_pred = out['grid'].detach().argmax(-1)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0) + (v.detach() if a.sync_fix else float(v.detach())) / a.unroll
    return total


def save_checkpoint(a, model, opt, cfg, it, b, amp):
    ck = dict(model=model.state_dict(), cfg=cfg.__dict__, iter=it, opt=opt.state_dict(), args=vars(a))
    torch.save(ck, os.path.join(a.out, 'ckpt.pt'))
    torch.save(ck, os.path.join(a.out, f'ckpt_{it:06d}.pt'))
    model.eval()
    with amp:
        save_rollout_png(os.path.join(a.out, f'rollout_{it:06d}.png'), model, b, model.init_memory(b['grid'].shape[0]),
                         a.unroll)
    model.train()
    print(f'saved checkpoint at {it}', flush=True)


def main():
    a = parse_args()
    rank, world, ddp = setup_distributed()
    master = rank == 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    win = load_data(a, rank, world)

    cfg = Config(d=a.d, layers=a.layers, heads=a.heads, mem=a.mem)
    model = WorldModel(cfg).to(device)
    opt = build_optimizer(model, a, device)
    ck, start_it = load_checkpoint(a), 0
    if ck is not None:
        model.load_state_dict(ck['model'])
        if 'opt' in ck:
            opt.load_state_dict(ck['opt'])
        start_it = ck.get('iter', 0)
        if master:
            print(f'resumed at iter {start_it}', flush=True)
    if master:
        print(f'params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M vocab {VOCAB}', flush=True)
    fwd = torch.compile(model) if a.compile else model
    if ddp:
        fwd = DDP(fwd, device_ids=[int(os.environ['LOCAL_RANK'])])
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else torch.autocast('cpu', enabled=False)

    batches = start_loader(win, a.batch)
    gen = torch.Generator(device=device).manual_seed(rank)
    agg, t_log = {}, time.time()
    end_it = min(a.iters, start_it + a.chunk) if a.chunk else a.iters
    for it in range(start_it, end_it):
        lr = lr_at(it, a)
        for g in opt.param_groups:
            g['lr'] = lr
        frac = it / a.iters
        p_ss, p_cor = a.ss_max * min(1, frac * 2), a.corrupt_max * min(1, frac * 3)
        b = to_torch(batches.get(), device)
        with amp:
            total = unroll_loss(fwd, model, b, a, p_ss, p_cor, gen, device, agg)
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if master and (it + 1) % a.log_every == 0:
            sps = a.log_every * a.batch * a.unroll * world / (time.time() - t_log)
            values = {k: float(v) / a.log_every for k, v in agg.items()}
            loss, grad_norm = float(total.detach()), float(gn)
            parts = ' '.join(f'{k} {v:.4f}' for k, v in values.items())
            print(f'it {it + 1} loss {loss:.4f} lr {lr:.2e} gn {grad_norm:.2f} ss {p_ss:.2f} '
                  f'| {parts} | {sps:.0f} samples/s', flush=True)
            append_metrics(a.out, dict(kind='train', it=it + 1, time=time.time(), loss=loss,
                                       **values, lr=lr, grad_norm=grad_norm, ss_prob=p_ss, samples_per_sec=sps))
            agg, t_log = {}, time.time()
        if master and ((it + 1) % a.ckpt_every == 0 or it + 1 == end_it):
            save_checkpoint(a, model, opt, cfg, it + 1, b, amp)
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
