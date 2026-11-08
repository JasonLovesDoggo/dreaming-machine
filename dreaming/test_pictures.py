import unittest
from collections import Counter

import numpy as np
from numpy.random import RandomState

from . import pictures as P
from .driver import Driver
from .os_sim import MicroOS
from .pictures import cats
from .pictures.draw import BLACK, WHITE, YELLOW, RED, H, W
from .tiles import CHARSET


def render(name, seed):
    return P.generate(name, RandomState(seed))


def background_share(img):
    return np.unique(img, return_counts=True)[1].max() / img.size


class CatPictureTest(unittest.TestCase):
    def test_colour_adjectives_compose_with_cat_words(self):
        for name, words in [('ORANGE CAT.JPEG', ['ORANGE', 'CAT']), ('RED CAT.PNG', ['RED', 'CAT']),
                            ('RED CAT', ['RED', 'CAT']), ('ORANGECAT.GIF', ['ORANGE', 'CAT']),
                            ('BLACK AND WHITE CAT.PNG', ['BLACK', 'WHITE', 'CAT']),
                            ('GINGER KITTEN.JPG', ['GINGER', 'KITTEN']), ('WHITE_KITTY7.BMP', ['WHITE', 'KITTY'])]:
            with self.subTest(name=name):
                self.assertEqual(P.parse_words(name), [(words, 'cat')])
        self.assertEqual(P.parse_name('ORANGE.PNG'), ['orange'])
        self.assertEqual(P.parse_name('RED APPLE.PNG'), ['apple'])
        self.assertEqual(P.parse_name('CAT AND DOG.PNG'), ['cat', 'dog'])
        self.assertEqual(P.parse_name('CATERPILLAR.PNG'), ['caterpillar'])

    def test_colour_sets_cat_body(self):
        for name, colour in [('ORANGE CAT.JPEG', YELLOW), ('RED CAT.PNG', RED), ('BLACK CAT.PNG', BLACK),
                             ('WHITE KITTEN.GIF', WHITE)]:
            cond = cats.cond_from_words(P.parse_words(name)[0][0])
            for seed in range(20):
                (body, _, _), _, _ = cats.random_spec(RandomState(seed), cond)
                self.assertEqual(body, colour, (name, seed))
        cond = cats.cond_from_words(['BLACK', 'WHITE', 'CAT'])
        (body, pattern, pcol), _, _ = cats.random_spec(RandomState(0), cond)
        self.assertEqual((body, pattern, pcol), (BLACK, 'bicolor', WHITE))

    def test_deterministic_bounded_and_varied(self):
        for name in ['CAT.JPG', 'CAT.GIF', 'KITTY.PNG', 'TABBY.JPG', 'ORANGE CAT.JPEG', 'RED CAT.PNG', 'SIAMESE.JPG',
                     'SLEEPY CALICO.BMP', 'CAT AND DOG.PNG']:
            with self.subTest(name=name):
                imgs = [render(name, s) for s in range(50)]
                for s in (0, 17, 42):
                    np.testing.assert_array_equal(imgs[s], render(name, s))
                for img in imgs:
                    self.assertEqual(img.shape, (H, W))
                    self.assertLess(img.max(), 8)
                self.assertGreater(len({img.tobytes() for img in imgs}), 45)

    def test_single_cats_are_dense_on_plain_backgrounds(self):
        for name in ['CAT.PNG', 'ORANGE CAT.JPEG', 'BLACK KITTEN.GIF']:
            shares = [background_share(render(name, s)) for s in range(200)]
            self.assertLess(np.mean(shares), 0.7, name)  # the cat covers ~1/3 of the canvas on average
            self.assertLess(np.percentile(shares, 90), 0.8, name)

    def test_all_types_and_poses_render(self):
        for _, coat, pose, _ in cats.CAT_TYPES:
            for kitten in (False, True):
                spec = (cats.COATS[coat], pose, {'kitten'} if kitten else set())
                img = cats.render(RandomState(3), spec)
                self.assertEqual(img.shape, (H, W))


class CatDriverTest(unittest.TestCase):
    def names(self, mode, n=4000):
        d = Driver(MicroOS(5), 5, mode)
        return [d.make_name() for _ in range(n)]

    def test_names_fit_and_are_mostly_cats(self):
        for mode, lo, hi in [('mixed', 0.6, 0.75), ('cats', 0.93, 1.0)]:
            names = self.names(mode)
            for n in names:
                self.assertLessEqual(len(n), 20)
                self.assertTrue(all(ch in CHARSET for ch in n), n)
            imgs = [n for n in names if P.is_image(n)]
            share = sum('cat' in P.parse_name(n) for n in imgs) / len(imgs)
            self.assertTrue(lo <= share <= hi, (mode, share))
        colours = Counter(w for n in self.names('mixed') for kws, c in P.parse_words(n) if c == 'cat'
                          for w in kws if w in P.COLOR_WORDS)
        self.assertGreaterEqual(len(colours), 10)
        self.assertIn('ORANGE', colours)
        self.assertIn('RED', colours)

    def test_cats_mode_opens_cat_pictures(self):
        o = MicroOS(11)
        o.reset()
        d = Driver(o, 11, 'cats')
        opens = Counter()
        for _ in range(3000):
            obs = o.step(next(d.stream))
            d.observe_events(obs['events'])
            if 'ex_open_png' in obs['events']:
                opens['cat' in P.parse_name(o.pt_file)] += 1
        self.assertGreater(opens[True], 8)
        self.assertGreater(opens[True], 0.9 * sum(opens.values()))


if __name__ == '__main__':
    unittest.main()
