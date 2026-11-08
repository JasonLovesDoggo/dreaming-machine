"""Training metrics: reads <runs_dir>/<run>/metrics.jsonl, one JSON object per line.

Train rows: {"kind": "train", "it": 25, "loss": ..., "grid": ..., ...}; every other numeric top-level
field becomes a series. Eval rows: {"kind": "eval", "it": ..., "metrics": {app: dreaming.eval result}}.
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


def _num(v):
    """(value, nonfinite): finite float, or None for missing/non-numeric/non-finite values."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None, False
    v = float(v)
    return (v, False) if math.isfinite(v) else (None, True)


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
        it, _ = _num(row.get('it'))
        if it is None or it < 0:
            self.skipped += 1
            return
        it = int(it)
        kind = row.get('kind', 'train')
        if kind == 'eval':
            self._eval(it, row.get('metrics'))
        elif kind == 'train':
            if self.last_it is not None and it <= self.last_it:
                self.restarts += 1
            self.last_it = it
            self._train(it, row)
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

    def _eval(self, it, metrics):
        if not isinstance(metrics, dict):
            self.skipped += 1
            return
        apps = {}
        for app, m in list(metrics.items())[:MAX_FIELDS]:
            if not _name(app) or not isinstance(m, dict):
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
                for key, short in (('acc_per_step', 'acc'), ('copy_first_acc_per_step', 'copy')):
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
                if curves:
                    self.curves.setdefault(app, {})[str(horizon)] = dict(it=it, **curves)
            apps[app] = out
        self.evals.append((it, apps))
        if len(self.evals) > MAX_EVAL_ROWS:
            del self.evals[:len(self.evals) - MAX_EVAL_ROWS]
            self.truncated = True


def _thin(n, points):
    if n <= points:
        return range(n)
    step = n / points
    return sorted({int(i * step) for i in range(points)} | {n - 1})


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


def _metrics(info, parsed, points):
    state = parsed(info['path'])
    evals = _latest_per_it(state.evals)
    apps = sorted({app for _, a in evals for app in a})
    return dict(
        id=info['id'], bytes=info['bytes'], mtime=info['mtime'], last_it=state.last_it,
        skipped_lines=state.skipped, nonfinite_values=state.nonfinite, restarts=state.restarts,
        truncated=state.truncated, notes=dict(state.notes),
        train=_columns(state.train, points),
        eval=dict(apps=apps, rows=len(evals), points={app: _columns([(it, a[app]) for it, a in evals if app in a], points)
                                                     for app in apps},
                  latest_rollouts=copy.deepcopy(state.curves)))
