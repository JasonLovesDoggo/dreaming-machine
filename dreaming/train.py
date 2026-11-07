"""Train the world model. Single GPU: python -m dreaming.train --data data/local
Multi GPU: torchrun --nproc_per_node 8 -m dreaming.train --data /data/shards"""
import argparse
import glob
import math
import os
import threading
import queue
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .dataset import load_shards, OBS_KEYS
from .model import WorldModel, Config, compute_loss, decode, NT
from .tiles import VOCAB, PAL


class Windows:
    """Random length-(L+1) windows that stay inside one episode."""

    def __init__(self, data, L, rng):
        self.d, self.L, self.rng = data, L, rng
        ends = np.cumsum(data['ep_len'])
        starts = ends - data['ep_len']
        self.valid = np.concatenate([np.arange(s, e - L - 1) for s, e in zip(starts, ends)])
        self.ep_start = np.repeat(starts, data['ep_len'])
        g = data['grid'].reshape(len(data['grid']), -1)
        changed = np.zeros(len(g), np.float64)
        changed[1:] = (g[1:] != g[:-1]).sum(1)
        cs = np.concatenate([[0], np.cumsum(changed)])
        act = cs[self.valid + L + 1] - cs[self.valid + 1]
        w = 0.25 + np.minimum(act, 50)
        self.p = w / w.sum()

    def batch(self, B):
        s = self.rng.choice(self.valid, B, p=self.p)
        idx = s[:, None] + np.arange(self.L + 1)[None]
        out = {k: self.d[k][idx] for k in OBS_KEYS}
        out['action'] = self.d['action'][idx]
        prev = np.where(idx - 1 >= self.ep_start[idx], self.d['action'][np.maximum(idx - 1, 0), 2], 0)
        out['prev_btn'] = prev.astype(np.uint8)
        return out


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


def save_rollout_png(path, model, b, mem0, L):
    """Teacher-forced vs free-running rollout for the first sample, saved as an image grid."""
    from .os_sim import obs_to_pixels
    from PIL import Image
    rows = []
    with torch.no_grad():
        mem = mem0[:1]
        mem_tf = mem0[:1]
        free = None
        for i in range(L):
            x = {k: v[:1] for k, v in step_inputs(b, i).items()}
            out_tf, mem_tf = model(x, mem_tf)
            xf = dict(x)
            if free is not None:
                for k in OBS_KEYS:
                    xf[k] = free[k]
            out_f, mem = model(xf, mem)
            free = decode(out_f)
            tf = decode(out_tf)
            def px(o):
                o = {k: v[0].cpu().numpy() for k, v in o.items()}
                o['grid'] = o['grid'].reshape(24, 32)
                return obs_to_pixels(o)
            gt = {k: v[:1] for k, v in step_targets(b, i).items()}
            rows.append(np.concatenate([px(gt), px(tf), px(free)], 1))
    im = np.concatenate(rows[:: max(1, L // 6)], 0)
    Image.fromarray(PAL[im]).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/local')
    ap.add_argument('--out', default='runs/local')
    ap.add_argument('--max-steps-data', type=int, default=0)
    ap.add_argument('--iters', type=int, default=20000)
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
    a = ap.parse_args()

    ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if ddp:
        dist.init_process_group('nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    else:
        rank, world = 0, 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    master = rank == 0
    os.makedirs(a.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(a.data, '*.npz')))
    paths = paths[rank::world] if len(paths) >= world else paths
    t0 = time.time()
    data = load_shards(paths, a.max_steps_data or None)
    if master:
        print(f'loaded {len(data["action"])} steps/rank from {len(paths)} shards in {time.time() - t0:.0f}s', flush=True)
    win = Windows(data, a.unroll, np.random.default_rng(1234 + rank))

    cfg = Config(d=a.d, layers=a.layers, heads=a.heads, mem=a.mem)
    model = WorldModel(cfg).to(device)
    start_it = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck['model'])
        start_it = ck.get('iter', 0)
    if master:
        print(f'params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M vocab {VOCAB}', flush=True)
    fwd = torch.compile(model) if a.compile else model
    if ddp:
        fwd = DDP(fwd, device_ids=[int(os.environ['LOCAL_RANK'])])
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.05}, {'params': nodecay, 'weight_decay': 0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=device == 'cuda')
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else torch.autocast('cpu', enabled=False)

    q = queue.Queue(maxsize=4)
    def loader():
        while True:
            q.put(win.batch(a.batch))
    for _ in range(2):
        threading.Thread(target=loader, daemon=True).start()

    gen = torch.Generator(device=device).manual_seed(rank)
    tl = time.time()
    agg = {}
    for it in range(start_it, a.iters):
        lr = a.lr * min(1, (it + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / a.iters)))
        for g in opt.param_groups:
            g['lr'] = lr
        frac = it / a.iters
        p_ss = a.ss_max * min(1, frac * 2)
        p_cor = a.corrupt_max * min(1, frac * 3)
        b = to_torch(q.get(), device)
        B = b['grid'].shape[0]
        mem = (model.init_memory(B))
        total = 0
        prev_pred = None
        with amp:
            for i in range(a.unroll):
                g_in = b['grid'][:, i]
                if prev_pred is not None and p_ss > 0:
                    use = torch.rand(B, 1, device=device, generator=gen) < p_ss
                    g_in = torch.where(use, prev_pred, g_in)
                if p_cor > 0:
                    m = torch.rand(g_in.shape, device=device, generator=gen) < p_cor * torch.rand(B, 1, device=device, generator=gen)
                    g_in = torch.where(m, torch.randint(0, VOCAB, g_in.shape, device=device, generator=gen), g_in)
                x = step_inputs(b, i, g_in)
                out, mem = fwd(x, mem)
                y = step_targets(b, i)
                loss, parts = compute_loss(out, y, b['grid'][:, i])
                total = total + loss / a.unroll
                prev_pred = out['grid'].detach().argmax(-1)
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0) + float(v.detach()) / a.unroll
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if master and (it + 1) % a.log_every == 0:
            dt = time.time() - tl
            sps = a.log_every * a.batch * a.unroll * world / dt
            msg = ' '.join(f'{k} {v / a.log_every:.4f}' for k, v in agg.items())
            print(f'it {it + 1} loss {float(total):.4f} lr {lr:.2e} gn {float(gn):.2f} ss {p_ss:.2f} '
                  f'| {msg} | {sps:.0f} samples/s', flush=True)
            agg, tl = {}, time.time()
        if master and ((it + 1) % a.ckpt_every == 0 or it + 1 == a.iters):
            ck = dict(model=model.state_dict(), cfg=cfg.__dict__, iter=it + 1)
            torch.save(ck, os.path.join(a.out, 'ckpt.pt'))
            model.eval()
            with amp:
                save_rollout_png(os.path.join(a.out, f'rollout_{it + 1:06d}.png'), model, b, model.init_memory(B),
                                 a.unroll)
            model.train()
            print(f'saved checkpoint at {it + 1}', flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
