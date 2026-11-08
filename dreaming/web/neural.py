"""World-model inference and the checkpoint registry shared by all browser sessions."""
import copy
import glob
import json
import os
import numpy as np
from ..dataset import OBS_KEYS


def _numbered(directory):
    checkpoints = []
    for path in glob.glob(os.path.join(directory, 'ckpt_*.pt')):
        iteration = os.path.basename(path)[5:-3]
        if iteration.isdigit():
            checkpoints.append((int(iteration), path))
    return sorted(checkpoints)


class Neural:
    """One world model stepping one dreamed computer. Copies share weights but keep their own memory."""

    def __init__(self, path, device):
        import torch
        from ..model import WorldModel, Config
        from ..os_sim import NKEYS
        ck = torch.load(path, map_location=device)
        self.model = WorldModel(Config(**ck['cfg'])).to(device).eval()
        sd = dict(ck['model'])
        akey = sd.get('akey.weight')
        if akey is not None and akey.shape[0] < NKEYS:  # checkpoints from before new keys were appended
            sd['akey.weight'] = torch.cat([akey, akey.new_zeros(NKEYS - akey.shape[0], akey.shape[1])])
        self.model.load_state_dict(sd)
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
    """Primary run plus compatible published runs; keeps a single model in memory."""

    def __init__(self):
        self.dir, self.root, self.device, self.preferred = None, None, 'cpu', None
        self._loaded = {}

    def configure(self, ckpt_dir, device, name=None, run_root=None):
        self.dir, self.root, self.device = ckpt_dir, run_root, device
        self._loaded = {}
        names = self.names()
        self.preferred = name if name in names else None
        if self.current:
            self.load(self.current)

    def paths(self):
        paths = {}
        if self.dir and os.path.isdir(self.dir):
            numbered = [path for _, path in _numbered(self.dir)]
            primary = numbered or glob.glob(os.path.join(self.dir, 'ckpt.pt'))
            paths.update((os.path.basename(p), p) for p in primary)
        if self.root:
            for marker in sorted(glob.glob(os.path.join(self.root, '*', 'run.json'))):
                directory = os.path.dirname(marker)
                if self.dir and os.path.realpath(directory) == os.path.realpath(self.dir):
                    continue
                try:
                    with open(marker) as f:
                        metadata = json.load(f)
                    if metadata.get('atlas_version') != 2:
                        continue
                    latest = int(metadata['iter'])
                except (OSError, ValueError, TypeError, KeyError, AttributeError):
                    continue
                for iteration, path in _numbered(directory):
                    filename = os.path.basename(path)
                    if iteration <= latest:
                        paths[f'{os.path.basename(directory)}/{filename}'] = path
        return paths

    def names(self):
        return list(self.paths())

    def default_name(self, names=None):
        names = self.names() if names is None else names
        return self.preferred if self.preferred in names else (names or [None])[-1]

    @property
    def current(self):
        return self.default_name()

    def load(self, name):
        path = self.paths()[name]
        if name not in self._loaded:
            self._loaded = {name: Neural(path, self.device)}
        return self._loaded[name]

    def session(self, name=None):
        """A per-session copy of the named (default: current) checkpoint, or None when there is none."""
        name = name or self.current
        return copy.copy(self.load(name)) if name else None


CHECKPOINTS = Checkpoints()
