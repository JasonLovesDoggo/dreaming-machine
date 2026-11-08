"""Standalone simulator, deliberately independent of MicroOS and the model."""
import base64
import os
import uuid
from collections import OrderedDict
from threading import Lock
from typing import Literal
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from ..apps import flight as fl
from ..tiles import grid_to_pixels
from .images import png

router = APIRouter(prefix='/flight')
MAX_SESSIONS = 32
sessions: OrderedDict[str, fl.Flight] = OrderedDict()
lock = Lock()


class Step(BaseModel):
    session: str | None = None
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    key: Literal['', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'ENTER', 'W', 'S', 'A', 'D'] = ''
    reset: bool = False


@router.get('')
def page():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'static', 'flight.html'))


@router.post('/api/step')
def step(request: Step):
    with lock:
        sid = request.session
        if sid is None:
            sid = uuid.uuid4().hex
            sessions[sid] = fl.Flight(request.seed)
            if len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
        if sid not in sessions:
            raise HTTPException(404, 'Flight session expired; start a new session')
        sim = sessions[sid]
        sessions.move_to_end(sid)
        events = sim.step('ENTER' if request.reset else request.key)
        frame = base64.b64encode(png(grid_to_pixels(fl.render_tiles(sim))).body).decode('ascii')
        return dict(session=sid, frame='data:image/png;base64,' + frame,
                    status=f'{sim.status} | ALT {sim.alt:.0f} SPEED {sim.speed * 30:.0f} THROTTLE {sim.throttle:.0%}',
                    alt=sim.alt, speed=sim.speed, throttle=sim.throttle, crash=bool(sim.crash), events=events)
