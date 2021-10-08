"""Trains BC, GAIL and AIRL models on saved CartPole-v1 demonstrations."""

import pathlib
import pickle
import tempfile

import seals  # noqa: F401
import stable_baselines3 as sb3

from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util

# my imports
import numpy as np
from imitation.data.types import Trajectory
from env_example import Toy_Environment
from stable_baselines3.common.vec_env import DummyVecEnv


def tj_converter(tj: list):
    obs = [list(tr['state']) for tr in tj] + [tj[-1]['next_state']]
    acts = [list(tr['action']) for tr in tj]
    infos = None
    terminal = True
    return Trajectory(obs=obs, acts=acts, infos=infos, terminal=terminal)


# Load pickled test demonstrations.
# demostrations_path = "../tests/testdata/expert_models/cartpole_0/rollouts/final.pkl"
demostrations_path = r".\final.pkl"
with open(demostrations_path, "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

# Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
# This is a more general dataclass containing unordered
# (observation, actions, next_observation) transitions.
trajectories = [tj_converter(tj) for tj in trajectories]
transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env("seals/CartPole-v0", n_envs=1)
data_folder = r'D:\work\imitation\tests\testdata\expert_models\my_env'

venv = Toy_Environment(log_level=2)
venv = DummyVecEnv([lambda: venv])

tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

# Train AIRL on expert data.
airl_logger = logger.configure(tempdir_path / "AIRL/")
airl_trainer = airl.AIRL(
    venv=venv,
    demonstrations=transitions,
    demo_batch_size=32,
    gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
    allow_variable_horizon=True,
    custom_logger=airl_logger,
)
airl_trainer.train(total_timesteps=20480)
