import unittest

import numpy as np
import torch

from . import doom
from .model import decode
from .os_sim import STATE_VOCAB
from .tiles import ATLAS, VOCAB


class DecodeTest(unittest.TestCase):
    def logits(self):
        shapes = {
            'grid': (2, 3, VOCAB), 'cx': (2, 256), 'cy': (2, 256),
            'd_type': (2, doom.VW, 5), 'd_side': (2, doom.VW, 2),
            'd_h': (2, doom.VW, 256), 'd_u': (2, doom.VW, 16),
            's_kind': (2, doom.NSPR, 5), 's_x': (2, doom.NSPR, 256),
            's_size': (2, doom.NSPR, 256), 'm_gun': (2, 4),
            'm_vc': (2, 32), 'm_vr': (2, 32), 'state': (2, len(STATE_VOCAB), 32),
        }
        out = {key: torch.full(shape, -100.0) for key, shape in shapes.items()}
        for value in out.values():
            value[..., 1] = 100.0
        out['m_gun'][0, 0] = 200.0
        out['m_gun'][1, 2] = 200.0
        return out

    def test_inactive_doom_feedback_is_zero_but_active_doom_is_unchanged(self):
        for temperature in (0.0, 0.6):
            with self.subTest(temperature=temperature):
                out = self.logits()
                before = {key: value.clone() for key, value in out.items()}
                decoded = decode(out, temperature)
                for key in ('doom_cols', 'doom_spr', 'doom_meta'):
                    self.assertEqual(int(decoded[key][0].count_nonzero()), 0)
                for key in ('doom_cols', 'doom_spr'):
                    self.assertTrue(torch.all(decoded[key][1] == 1))
                self.assertEqual(decoded['doom_meta'][1].tolist(), [2, 1, 1])
                for key, value in out.items():
                    self.assertTrue(torch.equal(value, before[key]))

    def test_unused_atlas_ids_are_excluded_even_when_they_have_highest_logits(self):
        unused = int(np.setdiff1d(np.arange(VOCAB), ATLAS.used)[0])
        valid = ATLAS[('solid', 'desk')]
        for temperature in (0.0, 0.6):
            with self.subTest(temperature=temperature):
                out = self.logits()
                out['grid'].fill_(-100.0)
                out['grid'][..., unused] = 1000.0
                out['grid'][..., valid] = 100.0
                decoded = decode(out, temperature)
                self.assertTrue(torch.all(decoded['grid'] == valid))

    def test_sampled_tiles_are_always_valid(self):
        out = self.logits()
        out['grid'].zero_()
        decoded = decode(out, 1.0)
        self.assertEqual(tuple(decoded['grid'].shape), (2, 3))
        self.assertEqual(decoded['grid'].dtype, torch.long)
        self.assertTrue(np.isin(decoded['grid'].numpy(), ATLAS.used).all())
