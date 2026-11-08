"""The play page's websocket: the real simulator and the world model stepped in lockstep.

Each server message is [n_frames:u8][has_ids:u8] + n_frames * 256x192 palette bytes
(+ n_frames * 768 little-endian u16 tile ids when the client asked for them)."""
import json
import logging
import random
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..os_sim import MicroOS, obs_to_pixels
from .neural import CHECKPOINTS

router = APIRouter()
log = logging.getLogger(__name__)


def _seed():
    return random.randrange(10 ** 9)


class Session:
    def __init__(self):
        self.real = MicroOS(_seed())
        self.obs = self.real.reset()
        self.neural = CHECKPOINTS.session()
        self.sync()

    def sync(self):
        if self.neural:
            self.neural.sync(self.obs)

    def command(self, cmd, ckpt=None):
        if cmd == 'reset':
            self.obs = self.real.reset(_seed())
            self.sync()
        elif cmd == 'sync':
            self.sync()
        elif cmd == 'load' and ckpt in CHECKPOINTS.names():
            try:
                self.neural = CHECKPOINTS.session(ckpt)
                self.sync()
            except Exception:
                log.exception('could not load checkpoint %s', ckpt)

    def step(self, action, temperature):
        try:
            self.obs = self.real.step(action)
        except Exception:
            log.exception('simulator error; rebooting')
            self.obs = self.real.reset(_seed())
            self.sync()
        screens = [self.obs]
        if self.neural:
            screens.append(self.neural.step(action, temperature))
        return screens


def encode(screens, with_ids):
    parts = [bytes([len(screens), int(with_ids)])]
    parts += [obs_to_pixels(o).astype(np.uint8).tobytes() for o in screens]
    if with_ids:
        parts += [np.asarray(o['grid'], '<u2').tobytes() for o in screens]
    return b''.join(parts)


@router.websocket('/ws')
async def play_socket(sock: WebSocket):
    await sock.accept()
    session = Session()
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            session.command(msg.get('cmd'), msg.get('ckpt'))
            action = tuple(int(msg[k]) for k in ('x', 'y', 'btn', 'key'))
            screens = session.step(action, float(msg.get('temp', 0.0)))
            await sock.send_bytes(encode(screens, bool(msg.get('ids'))))
    except WebSocketDisconnect:
        pass
