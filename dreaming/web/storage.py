"""Serialize mounted-volume access: Modal exposes an empty mount during reload."""
import functools
import threading

volume_lock = threading.RLock()


def volume_read(fn):
    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        with volume_lock:
            return fn(*args, **kwargs)
    return guarded
