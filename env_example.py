import numpy as np
import gym
from collections import OrderedDict
from gym import spaces


def vectorize_obs(observation, add_batch_dim=True) -> np.ndarray:
    """Converts the default dict of observations in an observation vector."""
    vector = np.array(list(observation.values())).flatten()
    if add_batch_dim:
        return vector.reshape(1, -1)
    else:
        return vector


class Toy_Environment(gym.Env):
    def __init__(self,
                 log_level: int = 0,
                 name: str = 'Toy_Environment'
                 ):
        """Initializes the game environment."""
        self.obs_dim = 48
        self.n_observations = 16
        self.baseline = 100  # min number of state transitions for final state reward no be positive or 0

        self.log_level = log_level
        self.name = name
        self.save_period = 2

        # Internal vars
        self.exit_env = False
        self.done = False
        # self.user_input_action = np.ones(2)
        self.state: OrderedDict = None
        self.trajectory_saver = []

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,))
        self.action_space = spaces.multi_discrete.MultiDiscrete([3, 3])
        self.num_envs = 1  # ???

        # Helper attr
        self._step_idx = 0

    # mandatory by openai environment interface
    def reset(self):
        # Reset the settings and states
        self._step_idx = 0
        # self.user_input_action = np.ones(2)
        self.trajectory_saver = []
        self.state = OrderedDict()
        self.observe()
        self.done = False
        return vectorize_obs(self.state, add_batch_dim=False)

    # mandatory by openai environment interface
    def step(self, action: np.ndarray):
        # self.handle_pygame_input()
        self.update(action)
        reward = self.true_reward()
        info = {}
        return vectorize_obs(self.state), reward, self.done, info

    def true_reward(self):
        """Returns a hand-crafted reward.
        1 - for every successful state till termination
        on the last state returns (baseline - number of steps in trajectory)
        """
        if not self.done:
            # add 1 for every successful step
            reward = 1
        else:  # termination state
            # Creates a baseline for performance:
            # negative final reward in case agent hasn't reached some predefined
            # number of transactions e.g. hasn't survived long enough
            # and positive reward in the other case (number of steps above the baseline)
            reward = self._step_idx - self.baseline
            if self.log_level >= 1:
                print(f"LAST REWARD: {reward}")
        return reward

    def update(self, action):
        """Updates all the game objects and states.
        Checks the player/enemies collisions and updates their positions.
        Stores the transitions.
        """
        # Look for player collisions and observe the env. Save the trajectory if player has lost.
        self.update_observations(action)
        # self._player.update(action)
        # self._enemies.update()
        # Update env attrs
        self._step_idx += 1

    def detect_collisions(self):
        return True if np.random.random() < 0.1 else False

    def update_observations(self, action):
        if self.detect_collisions():
            # finish the current game
            self.done = True
        self.observe_store(action)

    def observe_store(self, action):
        # store observation every #save_period steps and on the terminal state
        if (self._step_idx % self.save_period) == 0 or self.done:
            if self.log_level >= 1:
                print(f"SAVING STATE...   STEP: {self._step_idx}")
            # Save the old state
            old_state = self.state.copy()
            # Calculate new state
            self.observe()
            # Save the trajectory
            if old_state:
                keys = ['state', 'action', 'next_state', 'done']
                vals = [vectorize_obs(old_state), action, vectorize_obs(self.state), self.done]
                tj = dict(zip(keys, vals))
                self.trajectory_saver.append(tj)

    # mandatory by openai environment interface
    def render(self, **kwargs):
        pass

    # TODO accelerate
    def observe(self):
        """Observes the env."""
        # Iterate through the angles (except the last one (360 deg))
        for i in range(self.n_observations):
            self.state[f'{i}'] = np.random.random(self.obs_dim // self.n_observations)


class RandomPolicy:
    def __init__(self,
                 observation_space,
                 action_space,
                 ):
        self.observation_space = observation_space
        self.action_space = action_space

    def action(self, *args) -> np.ndarray:
        action = np.array(self.action_space.sample())
        return action
