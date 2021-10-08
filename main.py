"""A test environment for the agent.
The goal is to avoid enemy agents for as long as possible.
Press "p" to pause, "q" to exit, "r" to restart.
Tweak game settings in the settings.py file.
"""

from env_example import Toy_Environment, RandomPolicy


def main():
    # Initialize the env
    env = Toy_Environment(log_level=1)

    # Policy
    policy = RandomPolicy(env.observation_space, env.action_space)

    # episode loop (enabling multiple "rounds" of the game/environment)
    while not env.exit_env:  # possible place to include other break conditions like time-limit
        # reset environment before starting an episode
        state = env.reset()

        # interaction loop (stepping through the timesteps of an episode/game round)
        while not (env.done or env.exit_env):
            action = policy.action(state)
            state, reward, done, info = env.step(action)
            env.render()

        print(f"EPISODE IS FINISHED ({len(env.trajectory_saver)} transitions)\nq")


if __name__ == "__main__":
    main()
