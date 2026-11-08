"""Best-effort JSONL metrics for training and evaluation."""
import json
import math
import os
import warnings

_warned = False


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def append_metrics(out_dir, record):
    global _warned
    try:
        line = json.dumps(_finite(record), allow_nan=False)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'metrics.jsonl'), 'a') as f:
            f.write(line + '\n')
            f.flush()
    except (OSError, TypeError, ValueError) as error:
        if not _warned:
            _warned = True
            warnings.warn(f'Could not append metrics: {error}', RuntimeWarning, stacklevel=2)
