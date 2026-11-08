import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from . import telemetry


class TelemetryTest(unittest.TestCase):
    def test_events(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, DREAMING_TELEMETRY_DIR=root):
            app = FastAPI()
            app.include_router(telemetry.router)
            client = TestClient(app, client=('198.51.100.1', 1234))
            self.assertEqual(client.get('/telemetry/summary').json()['sessions'], 0)
            event = dict(visitor='anonymous', session='session', page='/', kind='pageview',
                         duration=5, apps={'paint': 1}, actions={'mouse': 2}, fps=15, latency_ms=3)
            self.assertEqual(client.post('/telemetry/events', json=event).status_code, 202)
            event.update(kind='heartbeat', duration=12, apps={}, actions={})
            client.post('/telemetry/events', json=event)
            result = client.get('/telemetry/summary').json()
            self.assertEqual((result['visitors'], result['sessions'], result['total_duration']), (1, 1, 12))
            self.assertEqual(result['pages'], {'/': 1})
            self.assertEqual(result['apps'], {'paint': 1})
            self.assertEqual(result['actions'], {'mouse': 2})
            self.assertEqual(result['latency_ms']['avg'], 3)
            path = next(telemetry.directory().glob('*.jsonl'))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows[0]['ip_hash']), 16)
            self.assertNotIn('198.51.100.1', path.read_text())
            with path.open('a') as f:
                f.write('{broken\n')
            self.assertEqual(client.get('/telemetry/summary').json()['sessions'], 1)
            for invalid in ({'duration': -1}, {'fps': 'NaN'}, {'visitor': 'x' * 81}):
                self.assertEqual(client.post('/telemetry/events', json=event | invalid).status_code, 422)
