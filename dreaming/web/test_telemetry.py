import json
import os
import tempfile
import threading
import unittest
from unittest.mock import Mock, call, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from . import telemetry


class TelemetryTest(unittest.TestCase):
    def test_refresh_commits_dirty_rows_before_reload(self):
        volume = Mock()
        with patch.multiple(telemetry, _volume=None, _dirty=True):
            refresh = telemetry.configure_modal(volume)
            refresh()
            self.assertFalse(telemetry._dirty)
            self.assertEqual(volume.method_calls, [call.commit(), call.reload()])
            refresh()
            self.assertEqual(volume.method_calls, [call.commit(), call.reload(), call.reload()])

    def test_failed_commit_preserves_dirty_rows(self):
        volume = Mock()
        volume.commit.side_effect = OSError('volume unavailable')
        with patch.multiple(telemetry, _volume=volume, _dirty=True):
            with self.assertRaises(OSError):
                telemetry.refresh_modal()
            self.assertTrue(telemetry._dirty)
            volume.reload.assert_not_called()

    def test_writes_and_summary_wait_for_volume_reload(self):
        refreshing, release = threading.Event(), threading.Event()
        written, summarized = threading.Event(), threading.Event()
        volume = Mock()

        def reload():
            refreshing.set()
            if not release.wait(5):
                raise TimeoutError('test did not release reload')

        def write():
            telemetry.write(telemetry.Event(visitor='visitor', session='session', page='/'))
            written.set()

        def summary():
            telemetry.summary()
            summarized.set()

        volume.reload.side_effect = reload
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, DREAMING_TELEMETRY_DIR=root), \
                patch.multiple(telemetry, _volume=volume, _dirty=False, _started=True, _salt=None):
            refresh = threading.Thread(target=telemetry.refresh_modal)
            writer = threading.Thread(target=write)
            reader = threading.Thread(target=summary)
            refresh.start()
            try:
                self.assertTrue(refreshing.wait(2))
                writer.start()
                reader.start()
                self.assertFalse(written.wait(.05))
                self.assertFalse(summarized.wait(.05))
            finally:
                release.set()
                for thread in (refresh, writer, reader):
                    if thread.ident is not None:
                        thread.join(2)
                        self.assertFalse(thread.is_alive())
            self.assertTrue(written.is_set())
            self.assertTrue(summarized.is_set())
            self.assertEqual(telemetry.summary()['sessions'], 1)
            self.assertTrue(telemetry._dirty)

    def test_failed_write_is_not_reported_as_accepted(self):
        app = FastAPI()
        app.include_router(telemetry.router)
        root = Mock()
        root.mkdir.side_effect = PermissionError('volume reloading')
        with patch.object(telemetry, 'directory', return_value=root), self.assertLogs(telemetry.log, 'ERROR'):
            response = TestClient(app).post('/telemetry/events', json=dict(visitor='v', session='s', page='/'))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {'detail': 'Telemetry storage unavailable'})

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
