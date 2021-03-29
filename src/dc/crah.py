import numpy as np

class CRAH:
    def __init__(self, n_crah, air_vol_heatcap):
        self.n_crah = n_crah
        self.air_vol_heatcap = air_vol_heatcap

        self.min_temp = 18
        self.max_temp = 27
        self.min_flow = 0.1 # To avoid divide by zero
        self.max_flow = 2.1 
        # servers are 50.4 / 0.04 watt/flow
        # crah should probably be similar (maybe cheaper) so starting we set it to 2.5 * 50.4 / 0.04 = 2646
        # Assume it is twice as effective
        self.max_fan_power = 2646 / 2

    def reset(self, ambient_temp):
        self.flow = self.min_flow * np.ones(self.n_crah)
        self.temp_out = 22 * np.ones(self.n_crah)

        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

        # If Tamb < Tout compressor is off
        self.compressor_power = np.sum((ambient_temp > self.temp_out) * self.air_vol_heatcap * self.flow * (ambient_temp - self.temp_out))

    def update(self, temp_out, flow, temp_in, ambient_temp):
        # Maybe allow individual control?
        self.flow = flow * np.ones(self.n_crah)
        self.temp_out = temp_out * np.ones(self.n_crah)

        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

        # If Tamb < Tout compressor is off
        self.compressor_power = np.sum((ambient_temp > self.temp_out) * self.air_vol_heatcap * self.flow * (temp_in - self.temp_out))