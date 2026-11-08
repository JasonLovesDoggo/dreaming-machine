import unittest
from collections import Counter

import numpy as np

from .dataset import run_episode
from .driver import Driver
from .os_sim import MicroOS, KEY_ENTER, T


class DriverTest(unittest.TestCase):
    def run_actions(self, os_, driver, actions):
        events = []
        for action in actions:
            obs = os_.step(action)
            driver.observe_events(obs['events'])
            events.extend(obs['events'])
        return events

    def test_new_file_enter_commits_then_opens(self):
        for name, app in [('NEW.PNG', 'paint'), ('NEW.TXT', 'notepad')]:
            with self.subTest(name=name):
                os_ = MicroOS(0)
                os_.open_app('explorer', 'C:/PICS')
                driver = Driver(os_, 0)
                wc, wr = os_.pos['explorer']
                self.run_actions(os_, driver, driver.click((wc + 9) * T + 4, (wr + 1) * T + 4))
                self.run_actions(os_, driver, driver.type_text(name))
                events = self.run_actions(os_, driver, driver.press_key(KEY_ENTER))
                self.assertIn('ex_new_file', events)
                self.assertEqual(os_.focus, 'explorer')
                self.assertIsNone(os_.ex_naming)
                self.assertEqual(os_.fs[os_.ex_path][os_.ex_sel][0], name)
                self.assertNotIn(app, os_.z)
                events = self.run_actions(os_, driver, driver.press_key(KEY_ENTER))
                self.assertIn('ex_open_png' if app == 'paint' else 'ex_open_txt', events)
                self.assertEqual(os_.focus, app)

    def test_paint_repeats_are_real_noops(self):
        counts = Counter()
        for seed in range(32):
            os_ = MicroOS(seed)
            os_.open_app('paint')
            driver = Driver(os_, seed)
            for action in driver.paint_repeat():
                canvas, color, file = os_.pt_canvas.copy(), os_.pt_color, os_.pt_file
                saved = file is not None and np.array_equal(os_.pictures[file], canvas)
                obs = os_.step(action)
                driver.observe_events(obs['events'])
                events = obs['events']
                if 'paint_dot' in events and np.array_equal(canvas, os_.pt_canvas):
                    self.assertNotEqual(color, 0)
                    counts['dot'] += 1
                if 'paint_color' in events and color == os_.pt_color:
                    counts['tool'] += 1
                if 'paint_clear' in events and not canvas.any():
                    counts['clear'] += 1
                if 'paint_save' in events and saved:
                    self.assertEqual(file, os_.pt_file)
                    counts['save'] += 1
        self.assertEqual(set(counts), {'dot', 'tool', 'clear', 'save'})

    def test_repeat_save_with_full_folder_does_not_create_file(self):
        os_ = MicroOS(0)
        os_.open_app('paint')
        os_.fs['C:/PICS'] += [(f'EXTRA{i}.PNG', 'png') for i in range(5)]
        driver = Driver(os_, 0)  # seed 0 chooses Save
        self.run_actions(os_, driver, driver.paint_repeat())
        self.assertEqual(len(os_.fs['C:/PICS']), 9)
        self.assertIsNone(os_.pt_file)

    def test_delete_selects_a_file_not_a_folder(self):
        for seed in range(10):
            os_ = MicroOS(seed)
            os_.open_app('explorer', 'C:')
            driver = Driver(os_, seed)
            events = self.run_actions(os_, driver, driver.delete_file())
            self.assertIn('ex_delete', events)
            self.assertEqual(os_.fs['C:'], [('PICS', 'folder'), ('DOCS', 'folder'), ('GAMES', 'folder')])
            self.assertIn(('README.TXT', 'txt'), os_.fs['TRASH'])
            self.assertEqual(os_.origin['README.TXT'], 'C:')

    def test_cardinal_drags_cover_sources_and_directions(self):
        directions, sources = set(), set()
        for seed in range(96):
            os_ = MicroOS(seed)
            os_.open_app('cards')
            os_.tab = [[[(12, 0), True], [(11, 2), True]], [[(5, 1), True]]] + [[] for _ in range(5)]
            os_.waste = [(0, 3)]
            os_.stock = []
            before = Counter(c for col in os_.tab for c, _ in col) + Counter(os_.waste)
            driver = Driver(os_, seed)
            held, dragged = [], False
            for action in driver.cards_cardinal():
                obs = os_.step(action)
                driver.observe_events(obs['events'])
                if 'cards_select' in obs['events']:
                    sel = os_.c_sel
                    sources.add('waste' if sel[0] == 'waste' else
                                'single' if sel[2] == len(os_.tab[sel[1]]) - 1 else 'stack')
                dragged |= 'cards_drag' in obs['events']
                if action[2]:
                    held.append(action[:2])
            self.assertTrue(dragged)
            x0, y0 = held[0]
            dx, dy = held[-1][0] - x0, held[-1][1] - y0
            self.assertTrue((dx == 0) != (dy == 0))
            self.assertTrue(all(x == x0 for x, y in held) if dx == 0 else all(y == y0 for x, y in held))
            directions.add('right' if dx > 0 else 'left' if dx < 0 else 'down' if dy > 0 else 'up')
            after = (Counter(c for col in os_.tab for c, _ in col) + Counter(os_.waste) +
                     Counter(c for pile in os_.found for c in pile))
            self.assertEqual(before, after)
        self.assertEqual(directions, {'left', 'right', 'up', 'down'})
        self.assertEqual(sources, {'single', 'stack', 'waste'})

    def test_seed_replays_identically(self):
        first, second = run_episode(17, 2000), run_episode(17, 2000)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)


if __name__ == '__main__':
    unittest.main()
