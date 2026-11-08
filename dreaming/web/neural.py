"""World-model inference and the checkpoint registry shared by all browser sessions."""
import copy
import glob
import os
import numpy as np
from ..dataset import OBS_KEYS


class Neural:
    """One world model stepping one dreamed computer. Copies share weights but keep their own memory."""

    def __init__(self, path, device):
        import torch
        from ..model import WorldModel, Config
        ck = torch.load(path, map_location=device)
        self.model = WorldModel(Config(**ck['cfg'])).to(device).eval()
        self.model.load_state_dict(ck['model'])
        self.device = device

    def sync(self, obs):
        self.obs = {k: np.array(obs[k]) for k in OBS_KEYS}
        self.mem = self.model.init_memory(1)
        self.prev_btn = 0

    def step(self, action, temperature=0.0):
        import torch
        from ..model import decode
        x = {k: torch.from_numpy(self.obs[k].astype(np.int64)).unsqueeze(0).to(self.device) for k in OBS_KEYS}
        x['grid'] = x['grid'].flatten(1)
        x['action'] = torch.tensor([action], dtype=torch.long, device=self.device)
        x['prev_btn'] = torch.tensor([self.prev_btn], device=self.device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device == 'cuda'):
            out, self.mem = self.model(x, self.mem)
        y = {k: v[0].cpu().numpy() for k, v in decode(out, temperature).items()}
        y['grid'] = y['grid'].reshape(24, 32)
        self.obs = {k: y[k] for k in OBS_KEYS}
        self.prev_btn = action[2]
        return self.obs


class Checkpoints:
    """Numbered checkpoints (ckpt_XXXXXX.pt) in one run directory; keeps a single model in memory."""

    def __init__(self):
        self.dir, self.device, self.current = None, 'cpu', None
        self._loaded = {}

    def configure(self, ckpt_dir, device, name=None):
        self.dir, self.device = ckpt_dir, device
        names = self.names()
        if names:
            self.current = name if name in names else names[-1]
            self.load(self.current)

    def names(self):
        if not self.dir or not os.path.isdir(self.dir):
            return []
        numbered = sorted(os.path.basename(f) for f in glob.glob(os.path.join(self.dir, 'ckpt_*.pt')))
        return numbered or [n for n in ['ckpt.pt'] if os.path.exists(os.path.join(self.dir, n))]

    def load(self, name):
        if name not in self._loaded:
            self._loaded = {name: Neural(os.path.join(self.dir, name), self.device)}
        return self._loaded[name]

    def session(self, name=None):
        """A per-session copy of the named (default: current) checkpoint, or None when there is none."""
        name = name or self.current
        return copy.copy(self.load(name)) if name else None


CHECKPOINTS = Checkpoints()
