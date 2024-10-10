import os
import shutil
from functools import partial
from os.path import basename
from timeit import default_timer as timer
from typing import Any, NamedTuple, Tuple

import hydra
import jax
import jax.numpy as jnp
from flax import struct
import imageio
import orbax
import optax
from flax.training.train_state import TrainState

from pcgrllm.utils.log_handler import TensorBoardLoggingHandler, WandbLoggingHandler, CSVLoggingHandler, \
    MultipleLoggingHandler
from purejaxrl.structures import RunnerState, Transition

import orbax.checkpoint as ocp
from jax.experimental.array_serialization.serialization import logger
from tensorboardX import SummaryWriter

import wandb  # wandb 추가
from conf.config import Config, TrainConfig
from envs.pcgrl_env import (gen_dummy_queued_state, gen_dummy_queued_state_old, OldQueuedState)
from pcgrllm.utils.logger import get_wandb_name
from pcgrllm.validate_reward import read_file
from purejaxrl.experimental.s5.wrappers import LogWrapper, LLMRewardWrapper
from pcgrllm.utils.path_utils import (get_ckpt_dir, get_exp_dir, init_network, gymnax_pcgrl_make, init_config)

import logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))



def log_callback(metric, steps_prev_complete, config, writer, train_start_time):
    timesteps = metric["timestep"][metric["returned_episode"]] * config.n_envs
    return_values = metric["returned_episode_returns"][metric["returned_episode"]]

    if len(timesteps) > 0:
        t = timesteps[-1].item()
        ep_return_mean = return_values.mean()
        ep_return_max = return_values.max()
        ep_return_min = return_values.min()

        ep_length = (metric["returned_episode_lengths"]
                     [metric["returned_episode"]].mean())
        fps = (t - steps_prev_complete) / (timer() - train_start_time)


        # wandb logging if enabled
        metric = {
            "ep_return": ep_return_mean,
            "ep_return_max": ep_return_max,
            "ep_return_min": ep_return_min,
            "ep_length": ep_length,
            "fps": fps,
        }
        writer.log(metric, t)


        print(f"fps: {fps}")
        print(f"global step={t}; episodic return mean: {ep_return_mean} " + \
              f"max: {ep_return_max}, min: {ep_return_min}")



def make_train(config, restored_ckpt, checkpoint_manager):
    config.NUM_UPDATES = (config.total_timesteps // config.num_steps // config.n_envs)
    config.MINIBATCH_SIZE = (config.n_envs * config.num_steps // config.NUM_MINIBATCHES)
    env_r, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env = env_r

    if hasattr(config, 'reward_function_path') and config.reward_function_path is not None:
        logger.info(f"Train using reward function from {config.reward_function_path}")
        env = LLMRewardWrapper(env)
        reward_fn_str = read_file(config.reward_function_path)
        exec_scope = {}
        exec(reward_fn_str, exec_scope)
        reward_fn = exec_scope['compute_reward']
        env.set_reward_fn(reward_fn)

    env = LogWrapper(env)
    env_r.init_graphics()

    def linear_schedule(count):
        frac = (1.0 - (count // (config.NUM_MINIBATCHES * config.update_epochs)) / config.NUM_UPDATES)
        return config["LR"] * frac

    def train(rng, config: TrainConfig):
        train_start_time = timer()

        # TensorBoard writer


        # checkwriter = SummaryWriter(config.exp_dir)



        #
        # if config.wandb_key and config.wandb_project:
        #
        #     # get the dir name of the experiment
        #
        #
        #     wandb.login(key=config.wandb_key)
        #     wandb.init(project=config.wandb_project, name=get_wandb_name(config), save_code=True)
        #     wandb.config.update(dict(config))
        #     logger.info(f"Initialized wandb with project {config.wandb_project}")
        # else:
        #     logger.info("wandb not initialized")

        # INIT NETWORK
        network = init_network(env, env_params, config)
        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(env_params)
        network_params = network.init(_rng, init_x)

        if config.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.lr, eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.n_envs)
        dummy_queued_state = gen_dummy_queued_state(env)
        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        obsv, env_state = vmap_reset_fn(reset_rng, env_params, dummy_queued_state)

        rng, _rng = jax.random.split(rng)
        steps_prev_complete = 0
        runner_state = RunnerState(train_state, env_state, obsv, rng, update_i=0)

        if restored_ckpt is not None:
            steps_prev_complete = restored_ckpt['steps_prev_complete']
            runner_state = restored_ckpt['runner_state']
            steps_remaining = config.total_timesteps - steps_prev_complete
            config.NUM_UPDATES = int(steps_remaining // config.num_steps // config.n_envs)



        handler_classes = [TensorBoardLoggingHandler, WandbLoggingHandler, CSVLoggingHandler]
        multiple_handler = MultipleLoggingHandler(config=config, handler_classes=handler_classes, logger=logger)

        # Set the start time and previous steps
        multiple_handler.set_start_time(train_start_time)
        multiple_handler.set_steps_prev_complete(steps_prev_complete)

        # During training, call the log method

        _log_callback = partial(log_callback,
                                config=config,
                                writer=multiple_handler,
                                train_start_time=train_start_time,
                                steps_prev_complete=steps_prev_complete)

        def step_env_render(carry, _):
            rng_r, obs_r, env_state_r, network_params = carry
            rng_r, _rng_r = jax.random.split(rng_r)

            pi, value = network.apply(network_params, obs_r)
            action_r = pi.sample(seed=rng_r)

            rng_step = jax.random.split(_rng_r, config.n_render_eps)
            vmap_step_fn = jax.vmap(env_r.step, in_axes=(0, 0, 0, None))
            obs_r, env_state_r, reward_r, done_r, info_r = vmap_step_fn(rng_step, env_state_r, action_r, env_params)
            vmap_render_fn = jax.vmap(env_r.render, in_axes=(0,))
            frames = vmap_render_fn(env_state_r)

            return (rng_r, obs_r, env_state_r, network_params), (env_state_r, reward_r, done_r, info_r, frames)

        def init_checkpoint(runner_state):
            ckpt = {'runner_state': runner_state, 'step_i': 0}
            checkpoint_manager.save(0, args=ocp.args.StandardSave(ckpt))

        def save_checkpoint(runner_state, info, steps_prev_complete):
            timesteps = info["timestep"][info["returned_episode"]] * config.n_envs
            if len(timesteps) > 0:
                t = timesteps[-1].item()
                latest_ckpt_step = checkpoint_manager.latest_step()
                if (latest_ckpt_step is None or t - latest_ckpt_step >= config.ckpt_freq):
                    print(f"Saving checkpoint at step {t}")
                    ckpt = {'runner_state': runner_state, 'step_i': t}
                    checkpoint_manager.save(t, args=ocp.args.StandardSave(ckpt))

        def _update_step(runner_state, unused):
            def _env_step(runner_state: RunnerState, unused):
                train_state, env_state, last_obs, rng, update_i = (
                    runner_state.train_state, runner_state.env_state,
                    runner_state.last_obs, runner_state.rng, runner_state.update_i
                )

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.n_envs)
                vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))
                obsv, env_state, reward, done, info = vmap_step_fn(rng_step, env_state, action, env_params)
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                runner_state = RunnerState(train_state, env_state, obsv, rng, update_i=update_i)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            train_state, env_state, last_obs, rng = runner_state.train_state, runner_state.env_state, runner_state.last_obs, runner_state.rng
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config.GAMMA * next_value * (1 - done) - value
                    gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        loss_actor1 = ratio * gae
                        loss_actor2 = (jnp.clip(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * gae)
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config.VF_COEF * value_loss - config.ENT_COEF * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(lambda x: jnp.reshape(x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])), shuffled_batch)
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            jax.debug.callback(save_checkpoint, runner_state, metric, steps_prev_complete)
            jax.debug.callback(_log_callback, metric)

            runner_state = RunnerState(train_state, env_state, last_obs, rng, update_i=runner_state.update_i + 1)
            return runner_state, metric

        jax.debug.callback(init_checkpoint, runner_state)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config.NUM_UPDATES)

        return {"runner_state": runner_state, "metrics": metric}

    return lambda rng: train(rng, config)


def init_checkpointer(config: Config) -> Tuple[Any, dict]:
    rng = jax.random.PRNGKey(30)
    ckpt_dir = get_ckpt_dir(config)
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env = LLMRewardWrapper(env)
    env = LogWrapper(env)

    rng, _rng = jax.random.split(rng)
    network = init_network(env, env_params, config)
    init_x = env.gen_dummy_obs(env_params)
    network_params = network.init(_rng, init_x)
    tx = optax.chain(optax.clip_by_global_norm(config.MAX_GRAD_NORM), optax.adam(config.lr, eps=1e-5))
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.n_envs)
    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
    obsv, env_state = vmap_reset_fn(reset_rng, env_params, gen_dummy_queued_state(env))
    runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=obsv, rng=rng, update_i=0)
    target = {'runner_state': runner_state, 'step_i': 0}
    ckpt_dir = os.path.abspath(ckpt_dir)

    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(ckpt_dir, options=options)

    def try_load_ckpt(steps_prev_complete, target):
        runner_state = target['runner_state']
        try:
            restored_ckpt = checkpoint_manager.restore(steps_prev_complete, args=ocp.args.StandardRestore(target))
        except KeyError:
            runner_state = runner_state.replace(env_state=runner_state.env_state.replace(env_state=runner_state.env_state.env_state.replace(queued_state=gen_dummy_queued_state_old(env))))
            target = {'runner_state': runner_state, 'step_i': 0}
            restored_ckpt = checkpoint_manager.restore(steps_prev_complete, items=target)

        restored_ckpt['steps_prev_complete'] = steps_prev_complete
        if restored_ckpt is None:
            raise TypeError("Restored checkpoint is None")

        return restored_ckpt

    if checkpoint_manager.latest_step() is None:
        restored_ckpt = None
    else:
        ckpt_subdirs = os.listdir(ckpt_dir)
        ckpt_steps = [int(cs) for cs in ckpt_subdirs if cs.isdigit()]
        ckpt_steps.sort(reverse=True)
        for steps_prev_complete in ckpt_steps:
            try:
                restored_ckpt = try_load_ckpt(steps_prev_complete, target)
                if restored_ckpt is None:
                    raise TypeError("Restored checkpoint is None")
                break
            except TypeError as e:
                print(f"Failed to load checkpoint at step {steps_prev_complete}. Error: {e}")
                continue

    return checkpoint_manager, restored_ckpt


def main_chunk(config, rng, exp_dir):
    checkpoint_manager, restored_ckpt = init_checkpointer(config)
    if restored_ckpt is None:
        progress_csv_path = os.path.join(exp_dir, "progress.csv")
        assert not os.path.exists(progress_csv_path), "Progress csv already exists, but have no checkpoint to restore " +\
            "from. Run with `overwrite=True` to delete the progress csv."
        with open(os.path.join(exp_dir, "progress.csv"), "w") as f:
            f.write("timestep,ep_return\n")

    train_jit = jax.jit(make_train(config, restored_ckpt, checkpoint_manager))
    out = train_jit(rng)
    jax.block_until_ready(out)

    return out


@hydra.main(version_base=None, config_path='./conf', config_name='train_pcgrl')
def main(config: TrainConfig):
    if config.initialize is None or config.initialize:
        config = init_config(config)

    rng = jax.random.PRNGKey(config.seed)
    exp_dir = config.exp_dir
    logger.info(f'running experiment at {exp_dir}')

    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    if config.timestep_chunk_size != -1:
        n_chunks = config.total_timesteps // config.timestep_chunk_size
        for i in range(n_chunks):
            config.total_timesteps = config.timestep_chunk_size + (i * config.timestep_chunk_size)
            print(f"Running chunk {i+1}/{n_chunks}")
            out = main_chunk(config, rng, exp_dir)

    else:
        out = main_chunk(config, rng, exp_dir)


if __name__ == "__main__":
    main()
