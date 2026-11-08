import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from . import app, neural, play


class CheckpointsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.primary = self.root / 'main'
        self.primary.mkdir()
        (self.primary / 'ckpt_005000.pt').touch()
        patcher = patch.object(neural, 'Neural')
        self.load = patcher.start()
        self.addCleanup(patcher.stop)
        self.checkpoints = neural.Checkpoints()

    def publish(self, name='atlas_v2_first_5k', **metadata):
        run = self.root / name
        run.mkdir(exist_ok=True)
        (run / 'ckpt_005000.pt').touch()
        (run / 'run.json').write_text(json.dumps(dict(atlas_version=2, iter=5000) | metadata))
        return run

    def configure(self, **kwargs):
        self.checkpoints.configure(str(self.primary), 'cpu', run_root=str(self.root), **kwargs)

    def test_publish_becomes_default_without_replacing_primary_file(self):
        run = self.publish()
        self.configure()
        self.assertEqual(self.checkpoints.current, 'atlas_v2_first_5k/ckpt_005000.pt')
        self.assertEqual(self.checkpoints.names(), ['ckpt_005000.pt', 'atlas_v2_first_5k/ckpt_005000.pt'])
        original = (self.primary / 'ckpt_005000.pt').read_bytes()
        self.checkpoints.load('atlas_v2_first_5k/ckpt_005000.pt')
        self.load.assert_called_with(str(run / 'ckpt_005000.pt'), 'cpu')
        self.assertEqual((self.primary / 'ckpt_005000.pt').read_bytes(), original)

    def test_only_completed_compatible_runs_are_visible(self):
        self.publish('legacy_v1', atlas_version=1)
        unfinished = self.publish('unfinished')
        (unfinished / 'ckpt_010000.pt').touch()
        (unfinished / 'ckpt_bad.pt').touch()
        broken = self.publish('broken')
        (broken / 'run.json').write_text('{')
        self.publish('wrong_type')
        (self.root / 'wrong_type' / 'run.json').write_text('[]')
        unpublished = self.root / 'unpublished'
        unpublished.mkdir()
        (unpublished / 'ckpt_005000.pt').touch()
        self.configure()
        self.assertEqual(self.checkpoints.names(), ['ckpt_005000.pt', 'unfinished/ckpt_005000.pt'])
        with self.assertRaises(KeyError):
            self.checkpoints.load('../legacy_v1/ckpt_005000.pt')

    def test_existing_single_run_and_empty_configuration(self):
        self.publish()
        self.checkpoints.configure(str(self.primary), 'cpu')
        self.assertEqual(self.checkpoints.names(), ['ckpt_005000.pt'])
        os.remove(self.primary / 'ckpt_005000.pt')
        (self.primary / 'ckpt.pt').touch()
        self.checkpoints.configure(str(self.primary), 'cpu')
        self.assertEqual(self.checkpoints.current, 'ckpt.pt')
        os.remove(self.primary / 'ckpt.pt')
        self.checkpoints.configure(str(self.primary), 'cpu')
        self.assertIsNone(self.checkpoints.current)
        self.assertEqual(self.checkpoints._loaded, {})

    def test_config_lists_both_runs_and_requested_default(self):
        self.publish()
        name = 'atlas_v2_first_5k/ckpt_005000.pt'
        with patch.object(app, 'CHECKPOINTS', self.checkpoints):
            client = TestClient(app.create_app(ckpt_dir=str(self.primary), ckpt_root=str(self.root), ckpt_name=name))
            config = client.get('/config').json()
        self.assertEqual(config['current'], name)
        self.assertEqual(config['ckpts'], self.checkpoints.names())

    def test_new_completed_run_is_discovered_without_reconfiguration(self):
        self.configure()
        self.publish()
        self.assertIn('atlas_v2_first_5k/ckpt_005000.pt', self.checkpoints.names())
        self.assertEqual(self.checkpoints.current, 'atlas_v2_first_5k/ckpt_005000.pt')

    def test_numeric_order_and_publication_boundary(self):
        run = self.publish(iter=10000)
        for name in ('ckpt_9000.pt', 'ckpt_10000.pt', 'ckpt_11000.pt'):
            (run / name).touch()
        self.configure()
        self.assertEqual(self.checkpoints.names()[-2:], [
            'atlas_v2_first_5k/ckpt_9000.pt', 'atlas_v2_first_5k/ckpt_10000.pt'])
        self.assertEqual(self.checkpoints.current, 'atlas_v2_first_5k/ckpt_10000.pt')

    def test_new_publication_only_changes_new_sessions(self):
        self.publish('run_a')
        self.configure()
        with patch.object(play, 'CHECKPOINTS', self.checkpoints):
            automatic = play.Session()
            existing = play.Session()
            existing.command('load', 'ckpt_005000.pt')
            original_model = existing.neural
            self.publish('run_b')
            newest = play.Session()
            self.assertEqual(newest.checkpoint, 'run_b/ckpt_005000.pt')
            automatic.command(None)
            self.assertEqual(automatic.checkpoint, 'run_a/ckpt_005000.pt')
            existing.command(None)
            self.assertEqual(existing.checkpoint, 'ckpt_005000.pt')
            self.assertIs(existing.neural, original_model)

    def test_config_and_first_websocket_frame_use_refreshed_default(self):
        self.configure()
        self.load.return_value.step.return_value = play.MicroOS(123).reset()
        with patch.object(app, 'CHECKPOINTS', self.checkpoints), patch.object(play, 'CHECKPOINTS', self.checkpoints):
            client = TestClient(app.create_app(ckpt_refresh=self.publish))
            config = client.get('/config').json()
            expected = 'atlas_v2_first_5k/ckpt_005000.pt'
            self.assertEqual(config['current'], expected)
            self.assertEqual(config['ckpts'][-1], expected)
            with client.websocket_connect('/ws') as ws:
                ws.send_json(dict(x=0, y=0, btn=0, key=0, telemetry=True))
                self.assertEqual(ws.receive_bytes()[:2], b'\x02\x00')
                self.assertEqual(ws.receive_json()['checkpoint'], expected)
            self.load.assert_called_with(str(self.root / expected), 'cpu')

    def test_load_command_acknowledges_selected_model(self):
        run = self.publish()
        self.configure()
        name = 'atlas_v2_first_5k/ckpt_005000.pt'
        with patch.object(play, 'CHECKPOINTS', self.checkpoints):
            session = play.Session()
            session.command('load', name)
            self.assertEqual(session.checkpoint, name)
            self.load.assert_called_with(str(run / 'ckpt_005000.pt'), 'cpu')
            session.command('load', '../legacy_v1/ckpt.pt')
            self.assertEqual(session.checkpoint, name)

    def test_config_refreshes_the_mounted_volume(self):
        with patch.object(app, 'CHECKPOINTS', self.checkpoints):
            client = TestClient(app.create_app(ckpt_dir=str(self.primary), ckpt_root=str(self.root), ckpt_refresh=self.publish))
            self.assertIn('atlas_v2_first_5k/ckpt_005000.pt', client.get('/config').json()['ckpts'])
