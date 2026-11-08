import io
import numpy as np
from fastapi.responses import Response
from ..tiles import PAL

PALETTE = np.asarray(PAL, np.uint8)


def png(px, scale=1, headers=None):
    """Palette-index (or RGB) array -> PNG response, optionally upscaled with nearest-neighbour."""
    from PIL import Image
    rgb = PALETTE[px] if px.ndim == 2 else px
    im = Image.fromarray(rgb)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return Response(buf.getvalue(), media_type='image/png', headers=headers)
