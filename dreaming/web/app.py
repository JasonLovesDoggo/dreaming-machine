import os
import logging
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ..tiles import PAL, CHARSET
from ..os_sim import EXTRA_KEYS, KEY_EXTRA0
from . import atlas, data_viewer, flight, play, telemetry, training
from .neural import CHECKPOINTS
from .settings import SETTINGS
from .storage import volume_lock, volume_read

STATIC = os.path.join(os.path.dirname(__file__), 'static')
log = logging.getLogger(__name__)


def _page(name):
    return lambda: FileResponse(os.path.join(STATIC, name))


def create_app(data_dir=None, ckpt_dir=None, device='cpu', ckpt_name=None, ckpt_root=None, ckpt_refresh=None,
               runs_dir=None, metrics_refresh=None):
    if data_dir:
        SETTINGS.data_dir = data_dir
    if ckpt_dir:
        CHECKPOINTS.configure(ckpt_dir, device, ckpt_name, run_root=ckpt_root)
    if runs_dir or metrics_refresh:
        training.configure(runs_dir, metrics_refresh, guard=volume_lock)

    app = FastAPI(title='Dreaming Computer')
    app.add_middleware(GZipMiddleware, minimum_size=4096)
    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    for router in (play.router, data_viewer.router, atlas.router, telemetry.router, training.router, flight.router):
        app.include_router(router)
    app.add_api_route('/', _page('play.html'), include_in_schema=False)
    app.add_api_route('/data', _page('data.html'), include_in_schema=False)
    app.add_api_route('/sprites', _page('atlas.html'), include_in_schema=False)
    app.add_api_route('/training', _page('training.html'), include_in_schema=False)

    @app.get('/config')
    @volume_read
    def config():
        if ckpt_refresh:
            try:
                ckpt_refresh()
            except Exception:
                log.exception('could not refresh checkpoint volume')
        names = CHECKPOINTS.names()
        return JSONResponse(dict(pal=PAL.tolist(), charset=CHARSET, extra_keys=EXTRA_KEYS, key_extra0=KEY_EXTRA0, sections=atlas.sections_json(),
                                 ckpts=names, current=CHECKPOINTS.default_name(names)))

    return app
