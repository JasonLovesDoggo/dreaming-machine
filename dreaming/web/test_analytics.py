import re
import shutil
import subprocess
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from .app import create_app

ROOT = Path(__file__).parent


class AnalyticsTest(unittest.TestCase):
    def test_page_and_navigation(self):
        with TestClient(create_app()) as client:
            response = client.get('/analytics')
            self.assertEqual(response.status_code, 200)
            for name in ('analytics.js', 'analytics_core.js'):
                self.assertIn(f'static/{name}', response.text)
                self.assertEqual(client.get(f'/static/{name}').status_code, 200)
            for page in ('/', '/training'):
                self.assertIn('href="analytics"', client.get(page).text)

    def test_only_aggregate_data_keys(self):
        for name in ('analytics.js', 'analytics_core.js'):
            source = (ROOT / 'static' / name).read_text()
            forbidden = r'(visitor|session|visitor_id|session_id|ip_hash|user_agent|text)'
            self.assertIsNone(re.search(r'\.' + forbidden + r'\b', source))
            self.assertIsNone(re.search(r'''\[\s*['"]''' + forbidden + r'''['"]\s*\]''', source))
            self.assertNotIn('innerHTML', source)
        source = (ROOT / 'static' / 'analytics.js').read_text()
        self.assertIn("fetch('telemetry/summary'", source)

    @unittest.skipIf(shutil.which('node') is None, 'node not installed')
    def test_core_helpers(self):
        out = subprocess.run(['node', str(ROOT / 'test_analytics_core.js')],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
