"""Contact sheets for eyeballing subjects: python -m dreaming.pictures.sheet --out sheets"""
import argparse
import os
import numpy as np
from PIL import Image, ImageDraw

from ..tiles import PAL, CANVAS_COLORS
from . import SUBJECTS, generate, cats, compose
from .draw import H, W, layer

_RGB = np.asarray(PAL, np.uint8)[CANVAS_COLORS]


def to_rgb(cv, px=8):
    return np.repeat(np.repeat(_RGB[cv], px, 0), px, 1)


def grid(rows, cols, px=6, label_h=12, pad=4, max_w=4096):
    """rows: list of (label, [canvas, ...]). One labelled row per entry, `cols` canvases per row, laid out in blocks."""
    cw, ch = W * px, H * px
    per_line = max(1, max_w // (cols * (cw + pad) + pad * 2))
    lines = (len(rows) + per_line - 1) // per_line
    bw = cols * (cw + pad) + pad * 2
    im = Image.new('RGB', (per_line * bw, lines * (ch + label_h + pad) + pad), (40, 40, 40))
    dr = ImageDraw.Draw(im)
    for i, (label, cvs) in enumerate(rows):
        x0, y0 = (i % per_line) * bw + pad, (i // per_line) * (ch + label_h + pad) + pad
        dr.text((x0, y0), label, fill=(255, 255, 255))
        for j, cv in enumerate(cvs):
            im.paste(Image.fromarray(to_rgb(cv, px)), (x0 + j * (cw + pad), y0 + label_h))
    return im


def subject_rows(seeds):
    rows = []
    for k, (fn, bg, words) in SUBJECTS.items():
        name = words.split()[0] + '.PNG'
        rows.append((f'{k}  ({name})', [generate(name, np.random.RandomState(sd)) for sd in seeds]))
    return rows


def cat_rows(seeds):
    rows = []
    for label, coat, pose, feats in cats.CAT_TYPES:
        cvs = []
        for sd in seeds:
            rng = np.random.RandomState(sd)
            cv, L = np.zeros((H, W), np.uint8), layer()
            cats.draw_cat(L, rng, W / 2 + rng.uniform(-2, 2), H / 2 + rng.uniform(-1, 1), rng.uniform(0.85, 1.1),
                          coat, pose, feats.split())
            cvs.append(compose(cv, L, rng, cats.CAT_BGS.get(pose, cats.DEFAULT_CAT_BGS)))
        rows.append((label, cvs))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='sheets')
    ap.add_argument('--seeds', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(a.seeds))
    grid(subject_rows(seeds), len(seeds)).save(os.path.join(a.out, 'subjects.png'))
    grid(cat_rows(seeds), len(seeds)).save(os.path.join(a.out, 'cats.png'))
    names = ['CAT.PNG', 'KITTEN.JPG', 'TABBY CAT.GIF', 'KITTY.PNG', 'SIAMESE.JPEG', 'BLACKCAT.BMP', 'CATNAP.PNG',
             'GINGER KITTEN.JPG', 'CAT AND DOG.PNG', 'CAT FISH.GIF']
    grid([(n, [generate(n, np.random.RandomState(sd)) for sd in range(8)]) for n in names], 8).save(
        os.path.join(a.out, 'cat_names.png'))


if __name__ == '__main__':
    main()
