"""World model: transformer over (tile grid + cursor + doom tokens + action + memory + state queries)."""
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tiles import VOCAB, GW, GH
from .os_sim import STATE_VOCAB, NKEYS
from . import doom as dm

NT = GW * GH
NS = len(STATE_VOCAB)
STATE_MAX = 32


@dataclass
class Config:
    d: int = 512
    layers: int = 12
    heads: int = 8
    mem: int = 32
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate='tanh'), nn.Linear(4 * d, d))

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x))


class WorldModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        E = lambda n: nn.Embedding(n, d)
        self.tile, self.tile_pos = E(VOCAB), nn.Parameter(torch.randn(NT, d) * 0.02)
        self.cx, self.cy = E(256), E(256)
        self.ax, self.ay, self.abtn, self.akey, self.aprev = E(256), E(256), E(2), E(NKEYS), E(2)
        self.dtype_, self.dside, self.dh, self.du = E(5), E(2), E(256), E(16)
        self.dcol_pos = nn.Parameter(torch.randn(dm.VW, d) * 0.02)
        self.skind, self.sx, self.ssize = E(5), E(256), E(256)
        self.spr_pos = nn.Parameter(torch.randn(dm.NSPR, d) * 0.02)
        self.mgun, self.mvc, self.mvr = E(4), E(32), E(32)
        self.mem_init = nn.Parameter(torch.randn(cfg.mem, d) * 0.02)
        self.mem_in = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.state_q = nn.Parameter(torch.randn(NS, d) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(8, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg.heads) for _ in range(cfg.layers)])
        self.ln_f = nn.LayerNorm(d)
        self.h_tile = nn.Linear(d, VOCAB)
        self.h_cur = nn.Linear(d, 512)
        self.h_dcol = nn.Linear(d, 5 + 2 + 256 + 16)
        self.h_spr = nn.Linear(d, 5 + 256 + 256)
        self.h_meta = nn.Linear(d, 4 + 32 + 32)
        self.h_state = nn.Linear(d, STATE_MAX)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def init_memory(self, B):
        return self.mem_init.unsqueeze(0).expand(B, -1, -1)

    def forward(self, x, mem):
        """x: grid (B,768) cursor (B,2) doom_cols (B,160,4) doom_spr (B,4,3) doom_meta (B,3) action (B,4) prev_btn (B,)"""
        B = x['grid'].shape[0]
        te = self.type_emb
        tiles = self.tile(x['grid']) + self.tile_pos + te[0]
        c = x['cursor']
        cur = (self.cx(c[:, 0]) + self.cy(c[:, 1]) + te[1]).unsqueeze(1)
        a = x['action']
        act = (self.ax(a[:, 0]) + self.ay(a[:, 1]) + self.abtn(a[:, 2]) + self.akey(a[:, 3]) +
               self.aprev(x['prev_btn']) + te[2]).unsqueeze(1)
        dc = x['doom_cols']
        dcol = self.dtype_(dc[..., 0]) + self.dside(dc[..., 1]) + self.dh(dc[..., 2]) + self.du(dc[..., 3]) + \
            self.dcol_pos + te[3]
        sp = x['doom_spr']
        spr = self.skind(sp[..., 0]) + self.sx(sp[..., 1]) + self.ssize(sp[..., 2]) + self.spr_pos + te[4]
        mt = x['doom_meta']
        meta = (self.mgun(mt[:, 0]) + self.mvc(mt[:, 1]) + self.mvr(mt[:, 2]) + te[5]).unsqueeze(1)
        memt = self.mem_in(mem) + te[6]
        sq = (self.state_q + te[7]).unsqueeze(0).expand(B, -1, -1)
        h = torch.cat([tiles, cur, act, dcol, spr, meta, memt, sq], 1)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        i = 0
        def take(n):
            nonlocal i
            s = h[:, i:i + n]
            i += n
            return s
        h_t, h_c, _, h_dc, h_sp, h_m, h_mem, h_s = take(NT), take(1), take(1), take(dm.VW), take(dm.NSPR), take(1), \
            take(self.cfg.mem), take(NS)
        cur_l = self.h_cur(h_c[:, 0])
        dcl = self.h_dcol(h_dc)
        spl = self.h_spr(h_sp)
        ml = self.h_meta(h_m[:, 0])
        out = dict(
            grid=self.h_tile(h_t),
            cx=cur_l[:, :256], cy=cur_l[:, 256:],
            d_type=dcl[..., :5], d_side=dcl[..., 5:7], d_h=dcl[..., 7:263], d_u=dcl[..., 263:],
            s_kind=spl[..., :5], s_x=spl[..., 5:261], s_size=spl[..., 261:],
            m_gun=ml[:, :4], m_vc=ml[:, 4:36], m_vr=ml[:, 36:],
            state=self.h_state(h_s),
        )
        return out, h_mem


def targets_from(y):
    return dict(grid=y['grid'], cx=y['cursor'][:, 0], cy=y['cursor'][:, 1],
                d_type=y['doom_cols'][..., 0], d_side=y['doom_cols'][..., 1], d_h=y['doom_cols'][..., 2],
                d_u=y['doom_cols'][..., 3], s_kind=y['doom_spr'][..., 0], s_x=y['doom_spr'][..., 1],
                s_size=y['doom_spr'][..., 2], m_gun=y['doom_meta'][:, 0], m_vc=y['doom_meta'][:, 1],
                m_vr=y['doom_meta'][:, 2], state=y['state'])


def compute_loss(out, y, in_grid, change_weight=10.0, state_weight=0.2):
    tg = targets_from(y)
    parts = {}
    ce = F.cross_entropy(out['grid'].flatten(0, 1).float(), tg['grid'].flatten(), reduction='none').view_as(tg['grid'])
    changed = (tg['grid'] != in_grid).float()
    w = 1 + (change_weight - 1) * changed
    parts['grid'] = (ce * w).sum() / w.sum()
    parts['cursor'] = F.cross_entropy(out['cx'].float(), tg['cx']) + F.cross_entropy(out['cy'].float(), tg['cy'])
    doom_on = (y['doom_meta'][:, 0] > 0).float()
    dl = 0
    for k in ['d_type', 'd_side', 'd_h', 'd_u', 's_kind', 's_x', 's_size']:
        l = F.cross_entropy(out[k].flatten(0, -2).float(), tg[k].flatten(), reduction='none').view(tg[k].shape[0], -1)
        dl = dl + l.mean(1)
    dl = (dl * doom_on).sum() / doom_on.sum().clamp(min=1)
    parts['doom'] = dl + sum(F.cross_entropy(out[k].float(), tg[k]) for k in ['m_gun', 'm_vc', 'm_vr'])
    parts['state'] = F.cross_entropy(out['state'].flatten(0, 1).float(), tg['state'].flatten())
    total = parts['grid'] + 0.2 * parts['cursor'] + 0.3 * parts['doom'] + state_weight * parts['state']
    with torch.no_grad():
        pred = out['grid'].argmax(-1)
        parts['acc_all'] = (pred == tg['grid']).float().mean()
        parts['acc_changed'] = ((pred == tg['grid']).float() * changed).sum() / changed.sum().clamp(min=1)
        parts['frac_changed'] = changed.mean()
    return total, parts


@torch.no_grad()
def decode(out, temperature=0.0):
    def pick(l):
        if temperature <= 0:
            return l.argmax(-1)
        return torch.distributions.Categorical(logits=l.float() / temperature).sample()
    return dict(
        grid=pick(out['grid']),
        cursor=torch.stack([out['cx'].argmax(-1), out['cy'].argmax(-1)], -1),
        doom_cols=torch.stack([pick(out[k]) for k in ['d_type', 'd_side', 'd_h', 'd_u']], -1),
        doom_spr=torch.stack([pick(out[k]) for k in ['s_kind', 's_x', 's_size']], -1),
        doom_meta=torch.stack([out[k].argmax(-1) for k in ['m_gun', 'm_vc', 'm_vr']], -1),
        state=out['state'].argmax(-1),
    )
