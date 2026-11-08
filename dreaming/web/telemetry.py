"""Small, anonymous JSONL analytics; no input text or raw addresses are retained."""
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import threading
import time
from collections import Counter

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from .storage import volume_lock

router = APIRouter(prefix='/telemetry')
log = logging.getLogger(__name__)
_lock = volume_lock
_started = False
_dirty = False
_salt = None
_volume = None
_writer = secrets.token_hex(8)


def configure_modal(volume):
    global _volume
    _volume = volume
    return refresh_modal


def refresh_modal():
    global _dirty
    with _lock:
        if _dirty:
            _volume.commit()
            _dirty = False
        _volume.reload()


class Event(BaseModel):
    visitor: str = Field(max_length=80)
    session: str = Field(max_length=80)
    page: str = Field(max_length=80)
    kind: str = Field(default='heartbeat', max_length=40)
    duration: float = Field(default=0, ge=0, le=604800, allow_inf_nan=False)
    actions: dict[str, int] = Field(default_factory=dict, max_length=64)
    apps: dict[str, int] = Field(default_factory=dict, max_length=32)
    checkpoint: str = Field(default='', max_length=160)
    fps: float | None = Field(default=None, ge=0, le=1000, allow_inf_nan=False)
    latency_ms: float | None = Field(default=None, ge=0, le=3600000, allow_inf_nan=False)


def directory():
    return Path(os.environ.get('DREAMING_TELEMETRY_DIR',
                               '/vol/telemetry' if _volume is not None else './runs/telemetry'))


def _committer():
    global _dirty
    while True:
        time.sleep(30)
        try:
            with _lock:
                if _dirty:
                    _volume.commit()
                    _dirty = False
        except Exception:
            log.exception('telemetry volume commit failed')


def write(event, client=None, user_agent=''):
    global _salt, _started, _dirty
    try:
        with _lock:
            root = directory()
            root.mkdir(parents=True, exist_ok=True)
            if _salt is None:
                salt_file = root / '.salt'
                try:
                    with salt_file.open('x') as f:
                        f.write(secrets.token_hex(32))
                except FileExistsError:
                    pass
                _salt = os.environ.get('DREAMING_TELEMETRY_SALT') or salt_file.read_text()
            row = event.model_dump()
            row['actions'] = {k[:80]: min(v, 1000000) for k, v in event.actions.items() if v > 0}
            row['apps'] = {k[:80]: min(v, 1000000) for k, v in event.apps.items() if v > 0}
            row.update(time=time.time(), user_agent=user_agent[:512],
                       ip_hash=hashlib.sha256((_salt + (client.host if client else '')).encode()).hexdigest()[:16])
            with (root / f'{time.strftime("%Y-%m-%d")}-{_writer}.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            _dirty = True
            if not _started and _volume is not None:
                _started = True
                threading.Thread(target=_committer, daemon=True).start()
        return True
    except Exception:
        log.exception('telemetry write failed')
        return False


@router.post('/events', status_code=202)
def ingest(event: Event, request: Request):
    if not write(event, request.client, request.headers.get('user-agent', '')):
        raise HTTPException(status_code=503, detail='Telemetry storage unavailable')
    return {'ok': True}


def _rows():
    with _lock:
        paths = list(directory().glob('*.jsonl'))
    for path in paths:
        offset = 0
        while True:
            with _lock:
                with path.open('rb') as f:
                    f.seek(offset)
                    chunk = f.read(1 << 20)
                    if chunk and not chunk.endswith(b'\n'):
                        chunk += f.readline(64 << 10)
                    offset = f.tell()
            if not chunk:
                break
            for line in chunk.splitlines():
                try:
                    yield json.loads(line)
                except (ValueError, UnicodeError):
                    continue


def _summary():
    visitors, sessions, pages = set(), {}, Counter()
    apps, actions, checkpoints = Counter(), Counter(), Counter()
    metrics = {'fps': [], 'latency_ms': []}
    for row in _rows():
        visitors.add(row['visitor'])
        key = (row['visitor'], row['session'], row['page'])
        sessions[key] = max(sessions.get(key, 0), row['duration'])
        if row['kind'] == 'pageview':
            pages[row['page']] += 1
        apps.update(row['apps'])
        actions.update(row['actions'])
        if row['checkpoint']:
            checkpoints[row['checkpoint']] += 1
        for name, values in metrics.items():
            if row.get(name) is not None:
                values.append(row[name])
    count = len({(v, s) for v, s, p in sessions})
    total = sum(sessions.values())
    return dict(visitors=len(visitors), sessions=count, total_duration=total,
                avg_duration=total / count if count else 0, pages=pages,
                apps=dict(apps.most_common()), actions=dict(actions.most_common()), checkpoints=checkpoints,
                **{k: dict(count=len(v), avg=sum(v) / len(v) if v else None,
                           min=min(v) if v else None, max=max(v) if v else None) for k, v in metrics.items()})


@router.get('/summary')
def summary():
    return _summary()
