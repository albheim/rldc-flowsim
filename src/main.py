import argparse

import ray
import ray.tune as tune
from ray.rllib.models import ModelCatalog

import loads 
from dc.dc import DCEnv
from loggerutils.loggingcallbacks import LoggingCallbacks

parser = argparse.ArgumentParser()
# Agent settings
parser.add_argument("--model", type=str, default="")
parser.add_argument("--crah_out_setpoint", type=float, default=22)
parser.add_argument("--crah_flow_setpoint", type=float, default=0.8)

# Env settings
parser.add_argument("--seed", type=int, default=37)
parser.add_argument("--avg_load", type=float, default=200)
parser.add_argument("--n_servers", type=int, default=40)#360)
parser.add_argument("--n_racks", type=int, default=1)#12)
parser.add_argument("--n_crah", type=int, default=1)#4)
parser.add_argument("--n_place", type=int, default=360) # How many to place load on, mostly for testing
parser.add_argument("--actions", nargs="+", default=["server", "crah_out", "crah_flow"])
parser.add_argument("--observations", nargs="+", default=["temp_out", "load", "job"])
parser.add_argument("--ambient", nargs=2, type=float, default=[20, 0])

# Training settings
parser.add_argument("--worker_seed", type=int, default=None) # Should make training completely reproducible, but might not work well with multiple workers in PPO
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--n_workers", type=int, default=1)
parser.add_argument("--pretrain_timesteps", type=int, default=0)
parser.add_argument("--stop_timesteps", type=int, default=500000)

args = parser.parse_args()

def trial_name_string(trial):
    name = str(trial)
    name += "_ACT_" + "_".join(args.actions)
    name += "_OBS_" + "_".join(args.observations)
    if args.tag != "":
        name += "_TAG_" + args.tag
    return name

# Job load
# avg_load = load_per_step / step_len * duration / servers => duration = step_len * avg_load * servers / load_per_step
dt = 1
load_per_step = 20
duration = dt * args.avg_load * args.n_servers / load_per_step
load_generator = loads.ConstantArrival(load=load_per_step, duration=duration)

# Ambient temp
temp_generator = loads.SinusTemperature(offset=args.ambient[0], amplitude=args.ambient[1])

# Init ray with all resources
# needs $ ray start --head --port 6379
ray.init(address="auto")

# Register env with ray
ray.tune.register_env("DCEnv", DCEnv)

config = {
    # Environment
    "env": "DCEnv",
    "env_config": {
        "dt": dt,
        "seed": args.seed,
        "n_servers": args.n_servers,
        "n_racks": args.n_racks,
        "n_crah": args.n_crah,
        "n_place": args.n_place,
        "load_generator": load_generator,
        "ambient_temp": temp_generator,
        "actions": args.actions,
        "observations": args.observations,
        "pretrain_timesteps": args.pretrain_timesteps,
        "crah_out_setpoint": args.crah_out_setpoint,
        "crah_flow_setpoint": args.crah_flow_setpoint,
    },

    # Model
    "model": {
        "custom_model": args.model,
        "custom_model_config": {
            "n_servers": args.n_servers,
        },
    },

    # Worker setup
    "num_workers": args.n_workers, # How many workers are spawned, data is aggregated from all
    "num_envs_per_worker": 1, # How many envs on a worker, can speed up if on gpu
    "num_gpus_per_worker": 0, # Use GPU if you have one
    "num_cpus_per_worker": 1, # Does this make any difference?
    "seed": args.worker_seed,

    # For logging (does soft_horizon do more, not sure...)
    "callbacks": LoggingCallbacks,
    "soft_horizon": True,
    "no_done_at_end": True,
    "horizon": 100, # Decides length of episodes
    "train_batch_size": 200 * args.n_workers, # Decides how often stuff is logged (will do min/max/avg over each episode datapoint)
    "rollout_fragment_length": 200,

    # Agent settings
    "vf_clip_param": 1000000.0, # Set this to be around the size of value function? Git issue about this not being good, just set high?

    # Data settings
    #"observation_filter": "MeanStdFilter", # Test this
    #"normalize_actions": True,
    #"checkpoint_at_end": True,
}

stop = {
    #"training_iteration": args.stop_iters,
    #"episode_reward_mean": args.stop_reward,
    "timesteps_total": args.stop_timesteps,
}

callbacks = [
]

results = tune.run(
    "PPO", 
    config=config, 
    callbacks=callbacks, 
    stop=stop, 
    trial_name_creator=trial_name_string,
    verbose=1,
    )
