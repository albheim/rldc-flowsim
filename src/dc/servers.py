import numpy as np
import heapq

class Servers:
    def __init__(self, n_servers, air_vol_heatcap, R):
        self.n_servers = n_servers
        self.air_vol_heatcap = air_vol_heatcap
        self.R = R

        self.idle_load = 50 
        self.max_load = 400  # W
        self.idle_temp_cpu = 35
        self.max_temp_cpu = 85  # C
        self.target_temp_cpu = 60
        self.min_flow = 0.001 # Have some small number since we dont want divide by zero
        self.max_flow = 0.04 # m3/s

        # TODO maybe find Ti in better way?
        # Ti is negative since a lower temp than ref means we should lower flow
        self.Ti = -10 * (self.max_temp_cpu - self.idle_temp_cpu) / (self.max_flow - self.min_flow)

        self.max_fan_power = 25.2 * 2 

    def reset(self, ambient_temp):
        self.delta_t = np.zeros(self.n_servers)
        self.temp_cpu = ambient_temp * np.ones(self.n_servers)
        self.flow = self.min_flow * np.ones(self.n_servers)
        self.load = self.idle_load * np.ones(self.n_servers)

        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

        self.running_jobs = []
        self.dropped_jobs = 0
        self.overheated_inlets = 0

    def update(self, time, dt, placement, load, duration, temp_in):
        # Update server in correct order
        new_temp_cpu = temp_in + self.R * self.load / self.flow
        delta_flow = dt / self.Ti * (self.target_temp_cpu - self.temp_cpu)
        new_flow = np.clip(self.flow + delta_flow, self.min_flow, self.max_flow)

        self.delta_t = self.load / (self.air_vol_heatcap * self.flow)

        self.temp_cpu = new_temp_cpu
        self.flow = new_flow
        self.overheated_inlets = np.sum(temp_in > 27)
        
        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

        self.dropped_jobs = 0
        if load == 0:
            pass # No job
        elif self.load[placement] + load <= self.max_load:
            self.load[placement] += load
            heapq.heappush(self.running_jobs, (time + duration, load, placement))
        else:
            self.dropped_jobs = 1
        while len(self.running_jobs) > 0 and self.running_jobs[0][0] <= time:
            _, load, placement = heapq.heappop(self.running_jobs)
            self.load[placement] -= load
