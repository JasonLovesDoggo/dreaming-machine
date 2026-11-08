import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ..dataset import run_episode
from . import data_viewer as viewer
from . import telemetry


class DataViewerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = os.path.join(self.temp.name, 'shards')
        setting = patch.object(viewer.SETTINGS, 'data_dir', self.root + '/')
        setting.start()
        self.addCleanup(setting.stop)
        viewer._index.clear()
        viewer._shard.cache_clear()
        viewer._ep_len.cache_clear()
        self.addCleanup(viewer._index.clear)
        self.addCleanup(viewer._shard.cache_clear)
        self.addCleanup(viewer._ep_len.cache_clear)
        self.write_shard(self.root, 2)
        self.write_shard(self.root + '_cats_dense_v1', 3)
        with open(os.path.join(self.root, 'sidecar.npz.json'), 'w') as f:
            json.dump({}, f)
        app = FastAPI()
        app.include_router(viewer.router)
        self.client = TestClient(app)

    @staticmethod
    def write_shard(root, frames):
        os.makedirs(root)
        data = run_episode(1, frames - 1)
        np.savez_compressed(os.path.join(root, 'same.npz'), **data, ep_len=np.array([frames]))

    def test_default_and_cache_isolation(self):
        self.assertEqual(self.client.get('/data/index').json()['shards'], ['same'])
        for dataset, frames in [('main', 2), ('cats_dense_v1', 3), ('main', 2)]:
            params = {} if dataset == 'main' else {'dataset': dataset}
            response = self.client.get('/data/shard', params=dict(name='same', **params))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), dict(name='same', dataset=dataset, ep_len=[frames],
                                                   frames=frames, episodes=1))
            response = self.client.get('/data/clip', params=dict(shard='same', start=0, n=32, **params))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.content), frames * 256 * 192)
            self.assertEqual(len(json.loads(response.headers['x-meta'])), frames)

    def test_dataset_and_shard_reads_wait_for_reload(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        responses = []
        volume = Mock()

        def reload():
            os.rename(self.root, self.root + '.hidden')
            entered.set()
            try:
                if not release.wait(5):
                    raise TimeoutError('reload was not released')
            finally:
                os.rename(self.root + '.hidden', self.root)

        def read():
            for path in ('/data/datasets', '/data/index', '/data/shard?name=same',
                         '/data/clip?shard=same&start=0'):
                responses.append(self.client.get(path))
            finished.set()

        volume.reload.side_effect = reload
        with patch.multiple(telemetry, _volume=volume, _dirty=False):
            refresh = threading.Thread(target=telemetry.refresh_modal)
            reader = threading.Thread(target=read)
            refresh.start()
            try:
                self.assertTrue(entered.wait(2))
                reader.start()
                self.assertFalse(finished.wait(.05))
                self.assertFalse(responses)
            finally:
                release.set()
                refresh.join(2)
                if reader.ident is not None:
                    reader.join(2)
            self.assertFalse(refresh.is_alive())
            self.assertFalse(reader.is_alive())
        self.assertEqual([r.status_code for r in responses], [200] * 4)
        self.assertTrue(responses[0].json()[0]['available'])
        self.assertEqual(responses[0].json()[0]['shards'], 1)
        self.assertEqual(responses[1].json()['shards'], ['same'])
        self.assertTrue(responses[1].json()['available'])

    def test_datasets_missing_and_all_roots(self):
        datasets = self.client.get('/data/datasets').json()
        self.assertEqual([d['id'] for d in datasets], ['main', 'cats_dense_v1', 'llm'])
        self.assertEqual([d['shards'] for d in datasets], [1, 1, 0])
        self.assertEqual([d['available'] for d in datasets], [True, True, False])
        self.assertEqual(datasets[1]['dir'], 'shards_cats_dense_v1')
        response = self.client.get('/data/index?dataset=llm')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['shards'], [])
        self.assertFalse(response.json()['available'])
        for endpoint, params in [('shard', {'name': 'same'}), ('clip', {'shard': 'same', 'start': 0})]:
            self.assertEqual(self.client.get('/data/' + endpoint, params=dict(dataset='llm', **params)).status_code, 404)
        self.write_shard(self.root + '_llm', 4)
        viewer._index.clear()
        self.assertEqual(self.client.get('/data/index?dataset=llm').json()['shards'], ['same'])
        self.assertEqual(self.client.get('/data/shard?dataset=llm&name=same').json()['frames'], 4)
        self.assertEqual(len(self.client.get('/data/clip?dataset=llm&shard=same&start=0').content), 4 * 256 * 192)

    def test_invalid_inputs_and_missing(self):
        for dataset in ['unknown', '../shards', '/etc/passwd']:
            for endpoint, params in [('index', {}), ('shard', {'name': 'same'}),
                                     ('clip', {'shard': 'same', 'start': 0})]:
                self.assertEqual(self.client.get('/data/' + endpoint, params=dict(dataset=dataset, **params)).status_code, 400)
        for name in ['../x', '..', 'a/b', '.hidden', '/etc/passwd', 'a..b', '']:
            self.assertEqual(self.client.get('/data/shard', params={'name': name}).status_code, 400)
            self.assertEqual(self.client.get('/data/clip', params={'shard': name, 'start': 0}).status_code, 400)
        self.assertEqual(self.client.get('/data/shard?name=missing').status_code, 404)
        self.assertEqual(self.client.get('/data/clip?shard=missing&start=0').status_code, 404)


class ViewerJavascriptTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'node is required for viewer client tests')
    def test_loading_errors_retry_and_stale_requests(self):
        script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const elements = new Map();
let renders = 0, mode = 'ok';
const calls = [];
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: '0', selectedIndex: 0, options: [], textContent: '', hidden: false,
    add(o) { this.options.push(o); if (this.options.length === 1) this.value = o.value; },
    set innerHTML(v) { this.options = []; this.value = ''; },
    getContext() { return {clearRect() {}, createImageData() { return {data: new Uint8Array(256*192*4)}; },
      putImageData() { renders++; }}; },
  });
  return elements.get(id);
}
const context = vm.createContext({console, Uint8Array, AbortController, URLSearchParams,
  document: {getElementById: element}, setInterval() {},
  Option: function(text, value = text) { this.text = text; this.value = value; },
  fetch: async url => {
    calls.push(url.split('?')[0]);
    if (mode === 'network') throw new Error('offline');
    if (mode === 'http') return {ok: false, status: 503, text: async () => 'unavailable'};
    return {ok: true, headers: {get: () => mode === 'meta' ? 'bad json' : JSON.stringify([['desktop', [0,0,0,0]]])},
      arrayBuffer: async () => new Uint8Array(mode === 'short' ? 1 : 256*192).buffer,
      json: async () => url.startsWith('data/datasets') ? [{id:'main',dir:'shards',shards:1,available:mode !== 'missing'}] :
        url.startsWith('config') ? {pal:[[0,0,0]]} :
        url.startsWith('data/index') ? {shards: mode === 'empty' ? [] : ['same'], dir:'shards'} :
        {ep_len:[1],frames:1}};
  },
});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const run = s => vm.runInContext(s, context);
const settle = async () => { await new Promise(setImmediate); };
(async () => {
  await settle();
  assert.equal(renders, 1);
  assert.deepEqual(calls.slice(0, 2), ['config', 'data/datasets']);
  mode = 'missing';
  await run('init()');
  assert.match(element('status').textContent, /No datasets available/);
  assert.equal(element('retry').hidden, false);
  mode = 'ok';
  await element('retry').onclick();
  assert.equal(element('status').textContent, '');
  const messages = [];
  Object.defineProperty(element('status'), 'textContent', {
    get() { return messages.at(-1) || ''; },
    set(value) { messages.push(value); },
  });
  await run('show()');
  assert.equal(messages.includes('loading…'), false);
  for (const failure of ['http', 'network', 'meta', 'short']) {
    mode = failure;
    await run('clips.clear(); show()');
    assert.match(element('status').textContent, /Error:/);
    assert.equal(element('retry').hidden, false);
    assert.equal(run('pending.size'), 0);
    mode = 'ok';
    await element('retry').onclick();
    assert.equal(element('status').textContent, '');
  }
  mode = 'empty';
  await run('pickDataset()');
  assert.equal(element('status').textContent, 'no shards in shards');
  mode = 'ok';
  await run('pickDataset()');
  const before = renders;
  let release;
  context.fetch = () => new Promise(resolve => { release = resolve; });
  const stale = run('clips.clear(); show()');
  run('change()');
  release({ok:true, headers:{get:()=>JSON.stringify([['desktop',[0,0,0,0]]])},
    arrayBuffer:async()=>new Uint8Array(256*192).buffer});
  await stale;
  assert.equal(renders, before);
  assert.equal(run('clips.size'), 0);
  const requests = [];
  context.fetch = async url => {
    requests.push(url);
    const start = Number(new URLSearchParams(url.split('?')[1]).get('start'));
    const count = Math.min(32, 33 - start);
    return {ok:true, headers:{get:()=>JSON.stringify(Array.from({length:count}, () => ['desktop',[0,0,0,0]]))},
      arrayBuffer:async()=>new Uint8Array(count*256*192).buffer};
  };
  await run('frames = 33; episodeLen = 33; clips.clear(); show()');
  await settle();
  assert.deepEqual(requests.map(url => new URLSearchParams(url.split('?')[1]).get('start')), ['0', '32']);
  assert.equal(run('clips.has("main:same:32")'), true);
})().catch(e => { console.error(e); process.exitCode = 1; });
"""
        result = subprocess.run(['node', '-e', script, str(Path(__file__).with_name('static') / 'data.js')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
