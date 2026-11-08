import unittest
import zlib
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from . import play
from .neural import Checkpoints


class PlayTest(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(play, 'CHECKPOINTS', Checkpoints())
        patcher.start()
        self.addCleanup(patcher.stop)
        seed = patch.object(play, '_seed', return_value=123)
        seed.start()
        self.addCleanup(seed.stop)
        app = FastAPI()
        app.include_router(play.router)
        self.client = TestClient(app)

    def test_original_and_compressed_formats(self):
        action = dict(x=100, y=80, btn=0, key=0)
        session = play.Session()
        expected = play.encode(session.step((100, 80, 0, 0), 0), False)
        for options in ({}, {'z': 0}, {'z': 1}):
            with self.client.websocket_connect('/ws') as ws:
                ws.send_json(action | options)
                payload = ws.receive_bytes()
                if options.get('z'):
                    self.assertEqual(payload[0], 255)
                    payload = zlib.decompress(payload[1:])
                self.assertEqual(payload, expected)
                self.assertEqual(payload[:2], b'\x01\x00')
                self.assertEqual(len(payload), 2 + 256 * 192)

    def test_pipelined_replies_with_ids(self):
        for compressed in (False, True):
            session = play.Session()
            actions = [(20, 30, 1, 0), (100, 90, 1, 0), (150, 110, 0, 0)]
            expected = [play.encode(session.step(a, 0), True) for a in actions]
            with self.client.websocket_connect('/ws') as ws:
                for x, y, btn, key in actions:
                    ws.send_json(dict(x=x, y=y, btn=btn, key=key, ids=True, z=compressed, telemetry=True))
                for frame in expected:
                    payload = ws.receive_bytes()
                    if compressed:
                        self.assertEqual(payload[0], 255)
                        payload = zlib.decompress(payload[1:])
                    self.assertEqual(payload, frame)
                    self.assertEqual(payload[:2], b'\x01\x01')
                    self.assertEqual(len(payload), 2 + 256 * 192 + 768 * 2)
                    telemetry = ws.receive_json()
                    self.assertGreaterEqual(telemetry['latency_ms'], 0)
                    self.assertIsNone(telemetry['checkpoint'])
