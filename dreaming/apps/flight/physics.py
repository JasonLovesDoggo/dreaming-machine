"""Fixed-step, bounded flight controls. Arrows fly; W/S throttle; A/D rudder; Enter resets."""
import math
import numpy as np
from .world import terrain


class Flight:
    def __init__(self, seed=0):
        self.seed = seed
        self.reset()

    def reset(self):
        self.x, self.y, self.alt = 0., -150., 10.
        self.speed, self.throttle, self.pitch, self.roll, self.heading = 0., 0., 0., 0., 0.
        self.crash = 0
        self.status = 'READY W THROTTLE UP ARROWS FLY'

    def step(self, key):
        if key == 'ENTER':
            self.reset()
            return ['flight_reset']
        if self.crash:
            self.crash -= 1
            if not self.crash:
                self.reset()
            return []
        self.throttle = float(np.clip(self.throttle + .04 * ((key == 'W') - (key == 'S')), 0, 1))
        self.pitch = float(np.clip(self.pitch * .98 + .025 * ((key == 'UP') - (key == 'DOWN')), -.65, .65))
        self.roll = float(np.clip(self.roll * .96 + .05 * ((key == 'RIGHT') - (key == 'LEFT')), -.8, .8))
        self.speed = max(0., self.speed + .08 * (self.throttle * 6 - self.speed) - self.pitch * .04)
        self.heading = (self.heading + self.roll * .025 + .02 * ((key == 'D') - (key == 'A'))) % (2 * math.pi)
        self.x += math.sin(self.heading) * self.speed
        self.y += math.cos(self.heading) * self.speed
        ground, color = terrain(self.x, self.y, self.seed)
        airborne = self.alt > float(ground) + 2.1
        vertical = self.speed * math.sin(self.pitch) - (max(0, 2.5 - self.speed) * .3 if airborne else 0)
        self.alt = min(350., self.alt + vertical)
        self.status = 'CRUISE' if airborne else 'RUNWAY - UP TO ROTATE'
        if airborne and self.speed < 2.5:
            self.status = 'STALL - LOWER NOSE'
        if self.alt <= float(ground) + 2:
            if color in (4, 5) and abs(vertical) < .8 and abs(self.roll) < .25:
                self.alt = float(ground) + 2
                self.status = 'LANDED' if airborne else self.status
            else:
                self.crash = 30
                self.status = 'CRASH - RESETTING'
                return ['flight_crash']
        return ['flight_fly'] if airborne else []
