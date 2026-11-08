"""Training metrics: reads <runs_dir>/<run>/metrics.jsonl, one JSON object per line.

Train rows: {"kind": "train", "it": 25, "loss": ..., "grid": ..., ...}; every other numeric top-level
field becomes a series. Eval rows: {"kind": "eval", "it": ..., "metrics": {app: dreaming.eval result}}, or
{"metrics": {"modes": {app: result}, <protocol metadata>...}}. Config rows: {"kind": "config", "config": {...}} carry
per-run metadata (frame-sample baseline, samples per update, checkpoint cadence, LR schedule) used for the summary.
Only run directories directly under the configured root are read, and only their metrics.jsonl
(never checkpoints). Files are parsed incrementally and bounded in bytes, rows and points."""
import copy
import json
import logging
import math
import os
import re
import stat
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix='/training')
log = logging.getLogger(__name__)

METRICS_FILE = 'metrics.jsonl'
RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')
FLAT_ROLLOUT = re.compile(r'roll(\d{1,6})_(acc|copy)_(mean|final)')
ROLLOUT_ACC = re.compile(r'rollout\d{1,6}_acc_(mean|final|at\d{1,6})')
MAX_RUNS = 200
MAX_BYTES = 16 << 20
MAX_LINE = 1 << 20
MAX_TRAIN_ROWS = 20000
MAX_EVAL_ROWS = 500
MAX_FIELDS = 64
MAX_NOTES = 16
MAX_STEPS = 1000
MAX_CACHED = 32
SCAN_TTL = 10.0
REFRESH_TTL = 60.0
POINTS = 2000
RECENT = 9
# The only config fields kept: (minimum, integer). Unknown keys are ignored, so config stays bounded.
CONFIG_FIELDS = {
    'frame_samples_base': (0, True), 'frame_samples_base_iter': (0, True), 'samples_per_update': (1, True),
    'world_size': (1, True), 'batch': (1, True), 'unroll': (1, True), 'log_every': (1, True),
    'checkpoint_every': (1, True), 'phase_end': (0, True), 'schedule_end': (0, True), 'restart_iter': (0, True),
    'warmup': (0, True), 'peak_lr': (0, False), 'min_lr_frac': (0, False),
}


def _num(v):
    """(value, nonfinite): finite float, or None for missing/non-numeric/non-finite values."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None, False
    v = float(v)
    return (v, False) if math.isfinite(v) else (None, True)


RESULT_KEYS = ('grid_ce', 'acc_all', 'acc_changed', 'copy_last_acc_all', 'rollouts')


def _is_result(m):
    return isinstance(m, dict) and any(k in m for k in RESULT_KEYS)


def _short(v):
    return f'{v:.4g}' if isinstance(v, float) else str(v)


def _name(k, limit=40):
    return isinstance(k, str) and 0 < len(k) <= limit


@dataclass
class _Parsed:
    ino: int = -1
    offset: int = 0
    train: list = field(default_factory=list)
    evals: list = field(default_factory=list)
    curves: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    skipped: int = 0
    nonfinite: int = 0
    restarts: int = 0
    truncated: bool = False
    last_it: int | None = None
    config: dict = field(default_factory=dict)
    config_it: int | None = None
    config_rejected: list = field(default_factory=list)
    sps: list = field(default_factory=list)
    update_secs: list = field(default_factory=list)
    prev: tuple | None = None

    def note(self, key, value):
        values = self.notes.get(key)
        if values is None:
            if len(self.notes) >= MAX_NOTES:
                return
            values = self.notes[key] = []
        if value not in values and len(values) < 4:
            values.append(value)

    def add(self, row):
        if not isinstance(row, dict):
            self.skipped += 1
            return
        if row.get('kind') == 'config':
            self._config(row)
            return
        it, _ = _num(row.get('it'))
        if it is None or it < 0:
            self.skipped += 1
            return
        it = int(it)
        kind = row.get('kind', 'train')
        if kind == 'eval':
            self._eval(it, row.get('metrics'))
        elif kind == 'train':
            restarted = self.last_it is not None and it <= self.last_it
            self.restarts += restarted
            self.last_it = it
            self._train(it, row)
            self._speed(it, self.train[-1][1], restarted)
        else:
            self.skipped += 1

    def _train(self, it, row):
        out = {}
        for k, v in row.items():
            if k in ('it', 'kind') or not _name(k):
                continue
            if isinstance(v, str):
                self.note(k, v[:80])
            elif isinstance(v, dict):
                text = ', '.join(f'{a}={b}' for a, b in v.items() if _name(a) and isinstance(b, (str, int, float)))
                self.note(k, text[:160])
            elif len(out) < MAX_FIELDS:
                value, bad = _num(v)
                self.nonfinite += bad
                if value is not None:
                    out[k] = value
        self.train.append((it, out))
        if len(self.train) > MAX_TRAIN_ROWS:
            del self.train[:len(self.train) - MAX_TRAIN_ROWS]
            self.truncated = True

    def _speed(self, it, values, restarted):
        """Recent throughput, skipping the first log after each (re)start: it includes torch.compile warm-up."""
        t, sps = values.get('time'), values.get('samples_per_sec')
        if self.prev is not None and not restarted:
            prev_it, prev_t = self.prev
            if sps is not None and sps > 0:
                self.sps = (self.sps + [sps])[-RECENT:]
            if t is not None and prev_t is not None and t > prev_t:
                self.update_secs = (self.update_secs + [(t - prev_t) / (it - prev_it)])[-RECENT:]
        self.prev = (it, t)

    def _config(self, row):
        cfg = row.get('config')
        if not isinstance(cfg, dict):
            self.skipped += 1
            return
        it, _ = _num(row.get('it'))
        for k, (low, integer) in CONFIG_FIELDS.items():
            if k not in cfg:
                continue
            value, _ = _num(cfg[k])
            if value is None or value < low or (integer and value != int(value)):
                if k not in self.config_rejected:
                    self.config_rejected.append(k)
                self.config.pop(k, None)
                continue
            self.config[k] = int(value) if integer else value
        self.config_it = int(it) if it is not None else self.config_it

    def _eval_note(self, it, key, value):
        """Protocol/provenance metadata next to the per-app results, e.g. eval_protocol, idle, eval_decode_commit."""
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            text = str(value)
        elif isinstance(value, dict):
            parts = []
            for k, v in value.items():
                if isinstance(v, dict):
                    v = ' '.join(f'{a}={_short(b)}' for a, b in v.items() if isinstance(b, (str, int, float)))
                elif isinstance(v, list):
                    v = ','.join(_short(b) for b in v[:8] if isinstance(b, (str, int, float)))
                elif not isinstance(v, (str, int, float)):
                    continue
                parts.append(f'{k}: {_short(v)}' if isinstance(v, (int, float)) else f'{k}: {v}')
            text = '; '.join(parts)
        else:
            return
        if text:
            self.note(f'eval {key}', f'@{it} {text}'[:240])

    def _eval(self, it, metrics):
        if not isinstance(metrics, dict):
            self.skipped += 1
            return
        modes = metrics.get('modes')
        results = modes if isinstance(modes, dict) else metrics
        for k, v in metrics.items():
            if (results is metrics and _is_result(v)) or (k == 'modes' and results is modes) or not _name(k):
                continue
            self._eval_note(it, k, v)
        apps = {}
        for app, m in list(results.items())[:MAX_FIELDS]:
            if not _name(app) or not _is_result(m):
                continue
            out = {}
            for k, v in m.items():
                value, bad = _num(v)
                self.nonfinite += bad
                if value is not None and _name(k):
                    flat = FLAT_ROLLOUT.fullmatch(k)
                    out[f'rollout{flat[1]}_{flat[2]}_{flat[3]}' if flat else k] = value
            rollouts = m.get('rollouts')
            for horizon, r in (rollouts.items() if isinstance(rollouts, dict) else ()):
                if not _name(str(horizon), 8) or not isinstance(r, dict):
                    continue
                curves = {}
                for key, short in (('acc_per_step', 'acc'), ('copy_first_acc_per_step', 'copy'), ('fp_change_rate_per_step', 'fp')):
                    steps = r.get(key)
                    if not isinstance(steps, list) or not steps:
                        continue
                    values = []
                    for v in steps[:MAX_STEPS]:
                        value, bad = _num(v)
                        self.nonfinite += bad
                        values.append(value)
                    curves[short] = values
                    finite = [v for v in values if v is not None]
                    if finite:
                        out[f'rollout{horizon}_{short}_mean'] = sum(finite) / len(finite)
                    if values[-1] is not None:
                        out[f'rollout{horizon}_{short}_final'] = values[-1]
                for key, short in (('acc_at_steps', 'acc'), ('copy_first_acc_at_steps', 'copy')):
                    at = r.get(key)
                    for step, v in (list(at.items())[:8] if isinstance(at, dict) else ()):
                        value, bad = _num(v)
                        self.nonfinite += bad
                        if value is not None and _name(str(step), 6):
                            out[f'rollout{horizon}_{short}_at{step}'] = value
                if curves:
                    self.curves.setdefault(app, {})[str(horizon)] = dict(it=it, **curves)
            for k in [k for k in out if ROLLOUT_ACC.fullmatch(k)]:
                base = out.get(k.replace('_acc_', '_copy_'))
                if base is not None:
                    out[k.replace('_acc_', '_gap_')] = out[k] - base
            apps[app] = out
        self.evals.append((it, apps))
        if len(self.evals) > MAX_EVAL_ROWS:
            del self.evals[:len(self.evals) - MAX_EVAL_ROWS]
            self.truncated = True


def _thin(n, points):
    if n <= points:
        return range(n)
    step = (n - 1) / (points - 1)
    return sorted({round(i * step) for i in range(points)})


def _latest_per_it(rows):
    """Sort by iteration, keeping the last-written row for an iteration repeated after a resume."""
    return sorted(dict(rows).items())


def _columns(rows, points):
    rows = _latest_per_it(rows)
    keep = [rows[i] for i in _thin(len(rows), points)]
    names = sorted({k for _, r in keep for k in r})
    return dict(it=[it for it, _ in keep], series={k: [r.get(k) for _, r in keep] for k in names}, rows=len(rows))


class Refresher:
    """Calls a volume reload in a background thread, at most once per ttl and only when the dashboard asks."""

    def __init__(self, fn=None, ttl=REFRESH_TTL, guard=None):
        self.fn, self.ttl, self.lock = fn, ttl, threading.Lock()
        self.guard = guard or threading.Lock()
        self.running, self.last_attempt, self.last_ok, self.last_error, self.last_error_at = False, 0.0, None, None, None
        self.on_done = None

    def poke(self):
        with self.lock:
            if self.fn is None or self.running or time.time() - self.last_attempt < self.ttl:
                return False
            self.running, self.last_attempt = True, time.time()
        threading.Thread(target=self._run, daemon=True, name='metrics-refresh').start()
        return True

    def _run(self):
        try:
            with self.guard:
                self.fn()
            self.last_ok, self.last_error = time.time(), None
        except Exception as e:
            self.last_error, self.last_error_at = f'{type(e).__name__}: {e}'[:300], time.time()
            log.warning('training metrics volume refresh failed: %s', self.last_error)
        finally:
            self.running = False
            if self.on_done:
                self.on_done()

    def status(self):
        return dict(enabled=self.fn is not None, running=self.running, ttl=self.ttl, last_ok=self.last_ok,
                    last_error=self.last_error, last_error_at=self.last_error_at)


class Store:
    def __init__(self, root=None, refresh=None, refresh_ttl=REFRESH_TTL):
        self.root = root or os.environ.get('DREAMING_RUNS_DIR', './runs')
        # Shared with the refresher: Modal refuses reload while a volume file is open and shows an empty
        # mount mid-reload, so scans/reads and reloads must never overlap.
        self.lock = threading.RLock()
        self.refresher = Refresher(refresh, refresh_ttl, guard=self.lock)
        self.refresher.on_done = self.invalidate
        self.scanned_at, self.scan, self.root_exists = 0.0, {}, False
        self.cache = OrderedDict()

    def invalidate(self):
        self.scanned_at = 0.0

    def runs(self, max_age=SCAN_TTL):
        with self.lock:
            if time.time() - self.scanned_at > max_age:
                self.scan, self.root_exists = self._scan()
                self.scanned_at = time.time()
                for path in [p for p in self.cache if p not in {r['path'] for r in self.scan.values()}]:
                    del self.cache[path]
            return self.scan

    def _scan(self):
        found = []
        try:
            entries = os.scandir(self.root)
        except OSError:
            return {}, False
        with entries:
            for e in entries:
                if not RUN_ID.fullmatch(e.name):
                    continue
                try:
                    if not e.is_dir(follow_symlinks=False):
                        continue
                    path = os.path.join(e.path, METRICS_FILE)
                    st = os.lstat(path)
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    found.append(dict(id=e.name, path=path, bytes=st.st_size, mtime=st.st_mtime))
        found.sort(key=lambda r: -r['mtime'])
        return {r['id']: r for r in found[:MAX_RUNS]}, True

    def lookup(self, run):
        if not isinstance(run, str) or not RUN_ID.fullmatch(run):
            raise HTTPException(404, 'no such run')
        info = self.runs().get(run) or self.runs(max_age=2.0).get(run)
        if info is None:
            raise HTTPException(404, 'no such run')
        return info

    def parsed(self, path):
        with self.lock:
            state = self.cache.pop(path, None) or _Parsed()
            try:
                fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            except OSError as e:
                raise HTTPException(404, f'cannot open metrics: {e.strerror}')
            with os.fdopen(fd, 'rb') as f:
                st = os.fstat(fd)
                if st.st_ino != state.ino or st.st_size < state.offset:
                    state = _Parsed(ino=st.st_ino)
                    if st.st_size > MAX_BYTES:
                        state.offset, state.truncated = st.st_size - MAX_BYTES, True
                        f.seek(state.offset)
                        state.offset += len(f.readline())
                while state.offset < st.st_size:
                    f.seek(state.offset)
                    chunk = f.read(min(MAX_BYTES, st.st_size - state.offset))
                    end = chunk.rfind(b'\n')
                    if end < 0:
                        if len(chunk) == MAX_BYTES:
                            state.skipped += 1
                            state.offset += len(chunk)
                            continue
                        break
                    for line in chunk[:end].split(b'\n'):
                        self._line(state, line)
                    state.offset += end + 1
            self.cache[path] = state
            while len(self.cache) > MAX_CACHED:
                self.cache.popitem(last=False)
            return state

    @staticmethod
    def _line(state, line):
        line = line.strip()
        if not line:
            return
        if len(line) > MAX_LINE:
            state.skipped += 1
            return
        try:
            state.add(json.loads(line))
        except (ValueError, UnicodeDecodeError, RecursionError):
            state.skipped += 1


STORE = Store()


def configure(runs_dir=None, refresh=None, refresh_ttl=REFRESH_TTL):
    global STORE
    STORE = Store(runs_dir, refresh, refresh_ttl)


@router.get('/runs')
def runs():
    refreshing = STORE.refresher.poke()
    found = STORE.runs()
    return dict(root=STORE.root, root_exists=STORE.root_exists, scanned_at=STORE.scanned_at,
                refresh=dict(STORE.refresher.status(), started=refreshing),
                runs=[dict(id=r['id'], bytes=r['bytes'], mtime=r['mtime']) for r in found.values()])


@router.get('/metrics')
def metrics(run: str = Query(max_length=128), points: int = Query(POINTS, ge=50, le=5000)):
    with STORE.lock:
        return _metrics(STORE.lookup(run), STORE.parsed, points)


def _median(xs):
    xs, n = sorted(xs), len(xs)
    return None if not n else xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _lr_phase(cfg, it):
    """Where `it` sits in the LR schedule the run's config row describes; None without that metadata."""
    start, warm, end = cfg.get('restart_iter'), cfg.get('warmup'), cfg.get('schedule_end')
    if it is None or start is None or warm is None or end is None or end <= start:
        return None
    peak, frac = cfg.get('peak_lr'), cfg.get('min_lr_frac')
    name = 'before restart' if it < start else 'warm-up' if it < start + warm else 'cosine decay' if it < end else 'floor'
    return dict(name=name, restart_iter=start, warmup_end=start + warm, schedule_end=end, peak_lr=peak,
                min_lr=peak * frac if peak is not None and frac is not None else None,
                progress=min(1.0, max(0.0, (it - start) / (end - start))))


def _summary(state, info, now):
    """Active-run numbers: only measured values or arithmetic on the run's own config row, otherwise None."""
    # The last-written row, not the highest iteration: after a resume from an earlier checkpoint this is the live cursor.
    cfg = state.config
    last_it, last = state.train[-1] if state.train else (None, {})
    last_time = last.get('time')
    sps = _median(state.sps)
    sec, source = _median(state.update_secs), 'row timestamps'
    if sec is None and sps and cfg.get('samples_per_update'):
        sec, source = cfg['samples_per_update'] / sps, 'config samples_per_update / samples/s'
    frames = None
    if last_it is not None and {'frame_samples_base', 'frame_samples_base_iter', 'samples_per_update'} <= cfg.keys() \
            and last_it >= cfg['frame_samples_base_iter']:
        frames = cfg['frame_samples_base'] + (last_it - cfg['frame_samples_base_iter']) * cfg['samples_per_update']
    since = last_time if last_time is not None else info['mtime']
    marks = []
    if last_it is not None:
        if cfg.get('checkpoint_every'):
            marks.append(('next checkpoint', (last_it // cfg['checkpoint_every'] + 1) * cfg['checkpoint_every']))
        marks += [(label, cfg[k]) for k, label in (('phase_end', 'phase end'), ('schedule_end', 'LR schedule end'))
                  if cfg.get(k) and cfg[k] > last_it]
    milestones = []
    for label, it in marks:
        eta = (it - last_it) * sec if sec is not None else None
        milestones.append(dict(label=label, it=it, updates=it - last_it, seconds=eta,
                               at=since + eta if eta is not None else None))
    return dict(last_it=last_it, last_time=last_time, file_mtime=info['mtime'], server_time=now,
                samples_per_sec=sps, samples_per_sec_logs=len(state.sps),
                sec_per_update=sec, sec_per_update_source=source if sec is not None else None,
                sec_per_update_logs=len(state.update_secs), frame_samples=frames, lr=last.get('lr'),
                lr_phase=_lr_phase(cfg, last_it), milestones=milestones,
                last_eval_it=max((it for it, _ in state.evals), default=None))


def _metrics(info, parsed, points):
    state = parsed(info['path'])
    evals = _latest_per_it(state.evals)
    apps = sorted({app for _, a in evals for app in a})
    config = None
    if state.config or state.config_rejected:
        config = dict(values=dict(state.config), it=state.config_it, rejected=list(state.config_rejected))
    return dict(
        id=info['id'], bytes=info['bytes'], mtime=info['mtime'], last_it=state.last_it,
        config=config, summary=_summary(state, info, time.time()),
        skipped_lines=state.skipped, nonfinite_values=state.nonfinite, restarts=state.restarts,
        truncated=state.truncated, notes=dict(state.notes),
        train=_columns(state.train, points),
        eval=dict(apps=apps, rows=len(evals), points={app: _columns([(it, a[app]) for it, a in evals if app in a], points)
                                                     for app in apps},
                  latest_rollouts=copy.deepcopy(state.curves)))
