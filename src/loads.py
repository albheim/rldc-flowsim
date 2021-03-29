import numpy as np

class ConstantArrival:
    def __init__(self, load, duration):
        self.load = load
        self.duration = duration
    def __call__(self, t):
        return (self.load, self.duration)
    def min_values(self):
        return (0, 0)
    def max_values(self):
        return (self.load, self.duration)

class RandomArrival:
    def __init__(self, load, duration, p, seed):
        self.load = load
        self.duration = duration
        self.p = p
        self.rng = np.random.default_rng(seed)
    def __call__(self, t):
        if self.rng.rand() < self.p:
            return (self.load, self.duration)
        else:
            return (0, 0)
    def min_values(self):
        return (0, 0)
    def max_values(self):
        return (self.load, self.duration)

class ConstantTemperature:
    def __init__(self, temp):
        self.temp = temp
    def __call__(self, t):
        return self.temp 
    def min_values(self):
        return 0
    def max_values(self):
        return self.temp

class SinusTemperature:
    def __init__(self, offset, amplitude):
        self.period = 24*60*60
        self.offset = offset
        self.amplitude = amplitude
    def __call__(self, t):
        return self.offset + self.amplitude * np.sin(2 * np.pi * t / self.period)
    def min_values(self):
        return self.offset - self.amplitude - 1
    def max_values(self):
        return self.offset + self.amplitude + 1