import copy
import os
from os.path import basename

import hydra
import imageio
import jax
from PIL import Image
import numpy as np

from conf.config import EnjoyConfig
from envs.pcgrl_env import PCGRLEnv, render_stats, gen_dummy_queued_state
from envs.probs.problem import get_loss, draw_solutions
from envs.solution import get_solution, get_solution_jit
from pcgrllm.task import TaskType
from purejaxrl.experimental.s5.wrappers import LossLogWrapper
from train import init_checkpointer
from utils import get_exp_dir, init_network, gymnax_pcgrl_make, init_config


import logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


@hydra.main(version_base=None, config_path='./conf', config_name='enjoy_pcgrl')
def main_rollout(enjoy_config: EnjoyConfig):
    if enjoy_config.initialize is None or enjoy_config.initialize:
        enjoy_config = init_config(enjoy_config)

    logger.info(f'Rollout config: {enjoy_config}')

    exp_dir = enjoy_config.exp_dir

    if not enjoy_config.random_agent:
        logger.info(f'Loading checkpoint from {exp_dir}')
        checkpoint_manager, restored_ckpt = init_checkpointer(enjoy_config)
        runner_state = restored_ckpt['runner_state']
        network_params = runner_state.train_state.params
    else:
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

    os.makedirs(enjoy_config._img_dir, exist_ok=True)
    os.makedirs(enjoy_config._numpy_dir, exist_ok=True)

    env, env_params = gymnax_pcgrl_make(enjoy_config.env_name, config=enjoy_config)
    env = LossLogWrapper(env)
    env.prob.init_graphics()
    network = init_network(env, env_params, enjoy_config)

    rng = jax.random.PRNGKey(enjoy_config.eval_seed)
    rng_reset = jax.random.split(rng, enjoy_config.n_envs)

    queued_state = gen_dummy_queued_state(env)

    obs, env_state = jax.vmap(env.reset, in_axes=(0, None, None))(
        rng_reset, env_params, queued_state
    )

    def step_env(carry, _):
        rng, obs, env_state = carry
        rng, rng_act = jax.random.split(rng)
        if enjoy_config.random_agent:
            action = env.action_space(env_params).sample(rng_act)[None, None, None, None]
        else:
            action = network.apply(network_params, obs)[0].sample(seed=rng_act)
        rng_step = jax.random.split(rng, enjoy_config.n_envs)
        obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            rng_step, env_state, action, env_params
        )
        frames = jax.vmap(env.render, in_axes=(0))(env_state.log_env_state.env_state)
        rng = jax.random.split(rng)[0]
        return (rng, obs, env_state), (env_state, reward, done, info, frames)

    step_env = jax.jit(step_env)

    _, (states, rewards, dones, infos, frames) = jax.lax.scan(
        step_env, (rng, obs, env_state), None,
        length=enjoy_config.n_eps * env.max_steps
    )

    env_maps = states.log_env_state.env_state.env_map

    cnt = 0
    for env_idx in range(enjoy_config.n_envs):
        for ep_idx in range(enjoy_config.n_eps):
            final_step = (ep_idx + 1) * env.max_steps - 2
            final_frame = frames[final_step, env_idx]
            final_level = env_maps[final_step, env_idx]

            if enjoy_config.task == TaskType.Scenario:
                solution = get_solution_jit(final_level)
                # change it to pil image
                final_frame = np.array(final_frame)
                final_frame = Image.fromarray(final_frame)
                final_frame = draw_solutions(final_frame, solution, env.prob.tile_size, np.array((1, 1)))
                final_frame = np.array(final_frame)

            # Save the final frame as PNG
            png_name = os.path.join(enjoy_config._img_dir, f"level_{cnt}.png")
            imageio.v3.imwrite(png_name, final_frame)

            # save numpy
            numpy_name = os.path.join(enjoy_config._numpy_dir, f"level_{cnt}.npy")
            np.save(numpy_name, env_maps[final_step, env_idx])

            cnt += 1


if __name__ == '__main__':
    main_rollout()
