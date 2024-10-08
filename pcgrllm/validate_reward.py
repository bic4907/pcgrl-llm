import numpy as np
from typing import Any

import hydra
import jax
import jax.numpy as jnp
from os import path

from conf.config import TrainLLMConfig
from envs.pcgrl_env import gen_dummy_queued_state
from pcgrllm.utils.logger import print_log
from purejaxrl.experimental.s5.wrappers import LogWrapper, LLMRewardWrapper
from pcgrllm.utils.path_utils import gymnax_pcgrl_make, init_config


# Get logger
import logging
from os.path import basename
logger = logging.getLogger(basename(__file__))
logger.setLevel(logging.DEBUG)


def read_file(file_path: str) -> Any:
    """Reads and returns the content of a file."""
    with open(file_path, 'r') as f:
        return f.read()


def run_validate(config: TrainLLMConfig):
    """Validates the environment setup and checks for NaN or Inf in rewards."""
    rng = jax.random.PRNGKey(30)

    # Set up the environment
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env = LLMRewardWrapper(env)
    env = LogWrapper(env)

    # Setup reward function
    if config.reward_function_path is None:
        config.reward_function_path = path.abspath(path.join(path.dirname(__file__), 'example', 'dummy_reward.py'))

    print_log(logger, f"Reward function path: {config.reward_function_path}", level=logging.INFO)
    reward_fn_str = read_file(config.reward_function_path)
    print_log(logger, f"\n{reward_fn_str}")

    exec_scope = {}
    exec(reward_fn_str, exec_scope)
    reward_fn = exec_scope['compute_reward']
    env.set_reward_fn(reward_fn)

    # Initialize environment state
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.n_envs)
    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))

    dummy_queued_state = gen_dummy_queued_state(env)
    obsv, env_state = vmap_reset_fn(reset_rng, env_params, dummy_queued_state)

    # Sample action and print
    rng_step = jax.random.split(_rng, config.n_envs)
    reward = jnp.zeros((config.n_envs, ), dtype=jnp.float32)

    # Prepare step function for lax.scan
    carry = (rng_step, env_state, reward, env_params)


    vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def step_fn(carry, _):
        """Step function for lax.scan that performs one step in the environment."""
        rng_step, env_state, reward, env_params = carry


        action = jax.vmap(env.rep.action_space.sample, in_axes=0)(rng_step)
        action = action.reshape(-1, 1, 1, 1)

        obsv, env_state, reward, done, info = vmap_step_fn(
            rng_step, env_state, action, env_params
        )

        carry = (rng_step, env_state, reward, env_params)  # Update carry with new state

        return carry, reward  # Return updated carry and current rewards

    _, rewards = jax.lax.scan(step_fn, carry, None, length=100)  # Run for 100 steps

    # Ensure final state is ready
    jax.block_until_ready(env_state)

    # Final reward check
    rewards = np.array(rewards)
    print_log(logger, f"Reward validation result: {rewards.shape}", level=logging.INFO)

    # Check for invalid rewards (NaN or Inf)
    if np.isnan(rewards).any():
        raise ValueError("Found NaN in the reward values")
    if np.isinf(rewards).any():
        raise ValueError("Found Inf in the reward values")

    print("Passed")

@hydra.main(version_base=None, config_path='../conf', config_name='train_pcgrllm')
def main(config: TrainLLMConfig) -> None:
    """Main entry point for validation."""
    if config.initialize is None or config.initialize:
        config = init_config(config)

    run_validate(config)


if __name__ == '__main__':
    main()
