from typing import Dict

import numpy as np

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

class LoggingCallbacks(DefaultCallbacks):
    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        pass
    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        pass
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        pass
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        env = base_env.get_unwrapped()[0]
        #print("Logging at env time {}".format(env.time))

        # Log server
        for i in range(env.n_servers):
            episode.custom_metrics[f"srv{i}/load"] = env.servers.load[i]
            episode.custom_metrics[f"srv{i}/temp_cpu"] = env.servers.temp_cpu[i]
            #episode.custom_metrics[f"srv{i}/flow"] = env.servers.flow[i]
            #episode.custom_metrics[f"srv{i}/temp_in"] = env.flowsim.server_temp_in[i]
            #episode.custom_metrics[f"srv{i}/temp_out"] = env.flowsim.server_temp_out[i]

        episode.custom_metrics["srv/max_temp_cpu"] = env.servers.temp_cpu.max()
        total_server_flow = np.sum(env.servers.flow)
        episode.custom_metrics["srv/server_total_flow"] = total_server_flow
        episode.custom_metrics["srv/overheated_inlets"] = env.servers.overheated_inlets
        episode.custom_metrics["srv/avg_temp_in"] = np.dot(env.flowsim.server_temp_in, env.servers.flow) / total_server_flow
        episode.custom_metrics["srv/avg_temp_out"] = np.dot(env.flowsim.server_temp_out, env.servers.flow) / total_server_flow
        episode.custom_metrics["srv/avg_temp_cpu"] = np.mean(env.servers.temp_cpu)
        episode.custom_metrics["srv/load_variance"] = np.var(env.servers.load)

        for i in range(env.n_crah):
            episode.custom_metrics[f"crah{i}/temp_in"] = env.flowsim.crah_temp_in[i]
            episode.custom_metrics[f"crah{i}/temp_out"] = env.crah.temp_out[i]
            episode.custom_metrics[f"crah{i}/flow"] = env.crah.flow[i]

        episode.custom_metrics["crah/crah_total_flow"] = np.sum(env.crah.flow)

        episode.custom_metrics["job/load"] = env.job[0]
        episode.custom_metrics["job/duration"] = env.job[1]

        # Should be 0 with the drop instead of delay?
        episode.custom_metrics["job/running"] = len(env.servers.running_jobs)
        episode.custom_metrics["job/dropped"] = env.servers.dropped_jobs

        episode.custom_metrics["power/server_fan"] = env.servers.fan_power
        episode.custom_metrics["power/crah_fan"] = env.crah.fan_power
        episode.custom_metrics["power/compressor"] = env.crah.compressor_power
        it_power = np.sum(env.servers.load)
        cooling_power = env.servers.fan_power + env.crah.fan_power + env.crah.compressor_power
        episode.custom_metrics["power/total_server_load"] = it_power
        episode.custom_metrics["power/PUE"] = (cooling_power + it_power) / it_power

        episode.custom_metrics["cost/energy"] = env.total_energy_cost
        episode.custom_metrics["cost/dropped"] = env.total_job_drop_cost
        episode.custom_metrics["cost/temp_cold_isle"] = env.total_overheat_cost
        
        episode.custom_metrics["other/ambient_temp"] = env.ambient_temp(env.time)