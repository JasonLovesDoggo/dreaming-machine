import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from . import training

STATIC = os.path.join(os.path.dirname(__file__), 'static')


def train_row(it, **extra):
    return dict(kind='train', it=it, loss=2.0 / it, grid=1.5 / it, cursor=0.1, doom=0.2, state=0.3, acc_all=0.9,
                acc_changed=0.2, frac_changed=0.05, lr=3e-4, grad_norm=1.2, ss_prob=0.1, samples_per_sec=900.0) | extra


def eval_row(it, acc=0.4):
    mode = dict(shard='heldout.npz', grid_ce=0.5, acc_all=0.95, acc_changed=acc, changed_tiles=120, copy_last_acc_all=0.93,
                rollouts={'30': dict(acc_per_step=[0.9, 0.8, 0.7], copy_first_acc_per_step=[0.9, 0.85, 0.8])})
    return dict(kind='eval', it=it, metrics=dict(mixed=mode, doom=dict(mode, acc_changed=0.1)))


class TrainingMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        training.configure(self.root)
        app = FastAPI()
        app.include_router(training.router)
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, run, rows, raw=''):
        os.makedirs(os.path.join(self.root, run), exist_ok=True)
        path = os.path.join(self.root, run, training.METRICS_FILE)
        with open(path, 'a') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)
            f.write(raw)
        training.STORE.invalidate()
        return path

    def metrics(self, run, **params):
        r = self.client.get('/training/metrics', params=dict(run=run, **params))
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_missing_root_is_empty_not_error(self):
        training.configure(os.path.join(self.root, 'absent'))
        body = self.client.get('/training/runs').json()
        self.assertEqual((body['runs'], body['root_exists'], body['refresh']['enabled']), ([], False, False))
        self.assertEqual(self.client.get('/training/runs').status_code, 200)

    def test_discovery_only_reads_metrics_files(self):
        self.write('atlas_v2_first_5k', [train_row(25)])
        self.write('audit_ab_b300.a', [train_row(25)])
        os.makedirs(os.path.join(self.root, 'no_metrics'))
        open(os.path.join(self.root, 'no_metrics', 'ckpt_000100.pt'), 'w').close()
        os.makedirs(os.path.join(self.root, '.hidden'))
        open(os.path.join(self.root, '.hidden', training.METRICS_FILE), 'w').close()
        outside = tempfile.mkdtemp()
        open(os.path.join(outside, training.METRICS_FILE), 'w').close()
        os.symlink(outside, os.path.join(self.root, 'linked_dir'))
        os.makedirs(os.path.join(self.root, 'linked_file'))
        os.symlink(os.path.join(outside, training.METRICS_FILE), os.path.join(self.root, 'linked_file', training.METRICS_FILE))
        opened = []
        real_open = os.open
        with patch.object(training.os, 'open', side_effect=lambda p, *a: opened.append(p) or real_open(p, *a)):
            ids = sorted(r['id'] for r in self.client.get('/training/runs').json()['runs'])
            self.assertEqual(ids, ['atlas_v2_first_5k', 'audit_ab_b300.a'])
            self.metrics('audit_ab_b300.a')
        self.assertEqual([os.path.basename(p) for p in opened], [training.METRICS_FILE])
        for bad in ('../x', '..', '/etc/passwd', 'no_metrics', 'linked_dir', '.hidden', 'a/b', 'missing'):
            self.assertEqual(self.client.get('/training/metrics', params=dict(run=bad)).status_code, 404, bad)

    def test_parses_train_eval_and_tolerates_bad_lines(self):
        hist = dict(source='historical_stdout', aggregation=dict(loss='last_update', parts='log_window_mean'))
        self.write('run', [train_row(25, **hist), train_row(50, **hist), eval_row(50)],
                   '{broken\n[1,2]\n{"kind":"train","it":"x"}\n{"kind":"other","it":1}\n'
                   '{"kind":"train","it":75,"loss":NaN,"grid":Infinity,"acc_all":true}\n{"kind":"train","it":100,"lo')
        d = self.metrics('run')
        self.assertEqual(d['train']['it'], [25, 50, 75])
        self.assertEqual(d['train']['series']['loss'], [0.08, 0.04, None])
        self.assertNotIn('acc_all', {k for k, v in d['train']['series'].items() if v[-1] is not None})
        self.assertEqual((d['skipped_lines'], d['nonfinite_values'], d['last_it']), (4, 2, 75))
        self.assertEqual(d['notes'], dict(source=['historical_stdout'], aggregation=['loss=last_update, parts=log_window_mean']))
        self.assertNotIn('source', d['train']['series'])
        self.assertEqual(d['eval']['apps'], ['doom', 'mixed'])
        mixed = d['eval']['points']['mixed']
        self.assertEqual((mixed['it'], mixed['series']['acc_changed']), ([50], [0.4]))
        self.assertAlmostEqual(mixed['series']['rollout30_acc_mean'][0], 0.8)
        self.assertEqual(mixed['series']['rollout30_copy_final'], [0.8])
        self.assertNotIn('shard', mixed['series'])
        self.assertEqual(d['eval']['latest_rollouts']['mixed']['30'], dict(it=50, acc=[0.9, 0.8, 0.7], copy=[0.9, 0.85, 0.8]))
        with open(os.path.join(self.root, 'run', training.METRICS_FILE), 'a') as f:
            f.write('ss":1}\n')
            f.write(json.dumps(train_row(125)) + '\n')
            f.write(json.dumps(dict(kind='eval', it=125, metrics=dict(mixed=dict(acc_changed=0.5, roll30_acc_final=0.6)))) + '\n')
        d = self.metrics('run')
        self.assertEqual((d['train']['it'][-2:], d['skipped_lines']), ([100, 125], 4))
        self.assertEqual(d['train']['series']['loss'][-2], 1)
        self.assertEqual(d['eval']['points']['mixed']['series']['rollout30_acc_final'], [0.7, 0.6])

    def test_resume_keeps_last_write_and_eval_order_is_not_a_restart(self):
        self.write('run', [train_row(25), train_row(50), eval_row(25), train_row(25, loss=9.0), train_row(50)])
        d = self.metrics('run')
        self.assertEqual((d['train']['it'], d['train']['series']['loss'][0], d['restarts']), ([25, 50], 9.0, 1))

    def test_points_and_bytes_are_bounded(self):
        self.write('run', [train_row(i * 25) for i in range(1, 301)])
        d = self.metrics('run', points=50)
        self.assertLessEqual(len(d['train']['it']), 51)
        self.assertEqual((d['train']['it'][0], d['train']['it'][-1], d['train']['rows']), (25, 7500, 300))
        self.assertEqual(self.client.get('/training/metrics', params=dict(run='run', points=10 ** 6)).status_code, 422)
        with patch.object(training, 'MAX_BYTES', 4096):
            training.configure(self.root)
            d = self.metrics('run')
        self.assertTrue(d['truncated'])
        self.assertEqual(d['train']['it'][-1], 7500)
        self.assertLess(d['train']['rows'], 300)

    def test_incremental_and_rewritten_files(self):
        path = self.write('run', [train_row(25)])
        self.assertEqual(self.metrics('run')['train']['it'], [25])
        self.write('run', [train_row(50)])
        self.assertEqual(self.metrics('run')['train']['it'], [25, 50])
        with open(path, 'w') as f:
            f.write(json.dumps(train_row(5)) + '\n')
        self.assertEqual(self.metrics('run')['train']['it'], [5])

    def test_volume_refresh_is_throttled_and_errors_are_reported(self):
        calls = []

        def refresh():
            calls.append(time.time())
            raise RuntimeError('there are open files')
        training.configure(self.root, refresh, refresh_ttl=3600)
        first = self.client.get('/training/runs').json()['refresh']
        self.assertTrue(first['started'])
        for _ in range(50):
            if not training.STORE.refresher.running:
                break
            time.sleep(0.01)
        second = self.client.get('/training/runs').json()['refresh']
        self.assertEqual((len(calls), second['started'], second['enabled']), (1, False, True))
        self.assertIn('open files', second['last_error'])

    def test_volume_reload_never_overlaps_reads_and_responses_are_snapshots(self):
        path = self.write('run_a', [train_row(25, source='historical_stdout'), eval_row(25)])
        seen = []

        def refresh():
            t = threading.Thread(target=lambda: seen.append(training.STORE.lock.acquire(blocking=False)))
            t.start()
            t.join()
        training.configure(self.root, refresh, refresh_ttl=3600)
        self.client.get('/training/runs')
        for _ in range(50):
            if not training.STORE.refresher.running:
                break
            time.sleep(0.01)
        self.assertEqual(seen, [False])
        body = training._metrics(training.STORE.lookup('run_a'), training.STORE.parsed, 100)
        state = training.STORE.parsed(path)
        self.assertIsNot(body['notes'], state.notes)
        self.assertIsNot(body['eval']['latest_rollouts'], state.curves)
        self.assertEqual(body['eval']['latest_rollouts'], state.curves)

    def test_page_footer_and_vendored_chart_library(self):
        for page in ('play.html', 'data.html', 'atlas.html'):
            with open(os.path.join(STATIC, page)) as f:
                self.assertIn('<a href="training">training metrics</a>', f.read())
        with open(os.path.join(STATIC, 'training.html')) as f:
            html = f.read()
        self.assertNotIn('://', html)
        for asset in ('vendor/uplot/uPlot.iife.min.js', 'vendor/uplot/uPlot.min.css', 'training.js'):
            self.assertIn(f'static/{asset}', html)
        for asset in ('vendor/uplot/uPlot.iife.min.js', 'vendor/uplot/uPlot.min.css', 'vendor/uplot/LICENSE', 'training.js'):
            self.assertTrue(os.path.isfile(os.path.join(STATIC, asset)), asset)

    def test_create_app_serves_training_page(self):
        try:
            from .app import create_app
        except ImportError as e:
            self.skipTest(f'web app dependencies missing: {e}')
        client = TestClient(create_app(runs_dir=self.root))
        self.assertIn('training metrics', client.get('/training').text)
        self.assertEqual(client.get('/training/runs').json()['root'], self.root)
        self.assertEqual(client.get('/static/vendor/uplot/uPlot.iife.min.js').status_code, 200)


if __name__ == '__main__':
    unittest.main()
