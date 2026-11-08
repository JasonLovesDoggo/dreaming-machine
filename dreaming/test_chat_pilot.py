import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'chat_pilot.py'


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_fresh_out_writes_shard(tmp_path):
    out = tmp_path / 'pilot'
    r = run('--out', out, '--episodes', 1, '--ep-len', 40, '--workers', 1, '--seed', 5)
    assert r.returncode == 0, r.stderr
    shard = out / 'shard_dreamer_pilot_005_00000.npz'
    assert int(np.load(shard)['ep_len'].sum()) == 41
    assert (out / 'shard_dreamer_pilot_005_00000.npz.json').exists()


def test_refuses_nonempty_out_without_touching_bytes(tmp_path):
    existing = tmp_path / 'shard_dreamer_pilot_005_00000.npz'
    existing.write_bytes(b'old pilot data')
    r = run('--out', tmp_path, '--episodes', 1, '--ep-len', 40, '--workers', 1, '--seed', 5)
    assert r.returncode != 0 and 'refusing to overwrite' in r.stderr
    assert existing.read_bytes() == b'old pilot data'
    assert sorted(p.name for p in tmp_path.iterdir()) == [existing.name]


def test_refuses_out_that_is_a_file(tmp_path):
    f = tmp_path / 'out'
    f.write_bytes(b'x')
    assert run('--out', f, '--episodes', 1, '--ep-len', 40, '--workers', 1).returncode != 0
    assert f.read_bytes() == b'x'


@pytest.mark.parametrize('flag', ['--episodes', '--ep-len', '--eps-per-shard', '--workers'])
@pytest.mark.parametrize('value', [0, -1])
def test_rejects_non_positive(tmp_path, flag, value):
    r = run('--out', tmp_path / 'pilot', flag, value)
    assert r.returncode == 2 and 'must be positive' in r.stderr
    assert not (tmp_path / 'pilot').exists()
