"""Fixed-step, bounded flight controls. Arrows fly; W/S throttle; A/D rudder; Enter resets."""
import math
import numpy as np
from .world import terrain, AIRFIELDS, RUNWAY, MARKING, ISLAND_HEIGHT


class Flight:
    def __init__(self, seed=0):
        self.seed = seed
        self.episode = 0
        self.reset()

    def reset(self):
        """Each reset spawns from a seeded episode RNG: on a runway, or airborne on approach."""
        rng = np.random.default_rng([self.seed, self.episode])
        self.episode += 1
        cx, cy = AIRFIELDS[int(rng.integers(len(AIRFIELDS)))]
        direction = 1 if rng.random() < .5 else -1
        self.heading = 0. if direction > 0 else math.pi
        self.pitch, self.roll = 0., 0.
        self.crash = 0
        if rng.random() < .7:
            self.x, self.y, self.alt = cx, cy - direction * 150., ISLAND_HEIGHT + 2
            self.speed, self.throttle = 0., 0.
            self.status = 'READY W THROTTLE UP ARROWS FLY'
        else:
            distance = float(rng.uniform(500, 900))
            self.x = cx + float(rng.uniform(-60, 60))
            self.y = cy - direction * distance
            self.alt = float(rng.uniform(50, 110))
            self.speed, self.throttle = 3.6, .6
            self.status = 'APPROACH'
        self.airborne = self.alt > ISLAND_HEIGHT + 2.1

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
        ground, material = terrain(self.x, self.y, self.seed)
        ground, material = float(ground), int(material)
        airborne = self.alt > ground + 2.1
        vertical = self.speed * math.sin(self.pitch) - (max(0, 2.5 - self.speed) * .3 if airborne else 0)
        self.alt = min(350., self.alt + vertical)
        self.status = 'CRUISE' if airborne else 'RUNWAY - UP TO ROTATE'
        if airborne and self.speed < 2.5:
            self.status = 'STALL - LOWER NOSE'
        events = []
        if self.alt <= ground + 2:
            gentle = abs(vertical) < .8 and abs(self.roll) < .25
            on_runway = material in (RUNWAY, MARKING)
            if not (gentle and (on_runway or (ground > 0 and self.speed < 2))):
                self.crash = 30
                self.status = 'CRASH - RESETTING'
                return ['flight_crash']
            self.alt = ground + 2
            if self.airborne:
                self.status = 'LANDED'
                events.append('flight_land')
            self.airborne = False
        elif not self.airborne:
            events.append('flight_takeoff')
            self.airborne = True
        return events + (['flight_fly'] if self.airborne else [])
