from enum import IntEnum
from typing import Optional

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.utils import Tiles


class Stats(IntEnum):
    pass


@struct.dataclass
class ProblemState:
    stats: Optional[chex.Array] = None
    ctrl_trgs: Optional[chex.Array] = None


def get_reward(stats, old_stats, stat_weights, stat_trgs, ctrl_threshes):
    """
    ctrl_threshes: A vector of thresholds for each metric. If the metric is within
        an interval of this size centered at its target value, it has 0 loss.
    """
    prev_loss = jnp.abs(stat_trgs - old_stats)
    prev_loss = jnp.clip(prev_loss - ctrl_threshes, 0)
    loss = jnp.abs(stat_trgs - stats)
    loss = jnp.clip(loss - ctrl_threshes, 0)
    reward = prev_loss - loss
    reward = jnp.where(stat_trgs == jnp.inf, stats - old_stats, reward)
    reward = jnp.where(stat_trgs == -jnp.inf, old_stats - stats, reward)
    reward *= stat_weights
    reward = jnp.sum(reward)
    return reward


def gen_ctrl_trgs(metric_bounds, rng):
    rng, _ = jax.random.split(rng)
    return jax.random.randint(rng, (len(metric_bounds),), metric_bounds[:, 0], metric_bounds[:, 1])


def gen_init_map(rng, tile_enum, map_shape, tile_probs):
    init_map = jax.random.choice(
        rng, len(tile_enum), shape=map_shape, p=tile_probs)
    return init_map


class Problem:
    tile_size = np.int8(16)
    stat_weights: chex.Array
    metrics_enum: IntEnum
    ctrl_metrics: chex.Array
    stat_trgs: chex.Array
    ctrl_threshes: chex.Array = None
    queued_ctrl_trgs: chex.Array = None

    def __init__(self, map_shape, ctrl_metrics):
        self.map_shape = map_shape
        self.metric_bounds = self.get_metric_bounds(map_shape)
        self.ctrl_metrics = np.array(ctrl_metrics, dtype=int)
        self.ctrl_metrics_mask = np.array([i in ctrl_metrics for i in range(len(self.stat_trgs))])

        if self.ctrl_threshes is None:
            self.ctrl_threshes = np.zeros(len(self.stat_trgs))

        # Dummy control observation to placate jax tree map during minibatch creation (FIXME?)
        self.ctrl_metric_obs_idxs = np.array([0]) if len(self.ctrl_metrics) == 0 else self.ctrl_metrics

        self.metric_names = [metric.name for metric in self.metrics_enum]

        self.queued_ctrl_trgs = jnp.zeros(len(self.metric_bounds))  # dummy value to placate jax
        self.has_queued_ctrl_trgs = False

    def gen_init_map(self, rng):
        return gen_init_map(rng, self.tile_enum, self.map_shape,
                               self.tile_probs)

    def get_metric_bounds(self, map_shape):
        raise NotImplementedError

    def get_stats(self, env_map: chex.Array, prob_state: ProblemState):
        raise NotImplementedError

    def queue_ctrl_trgs(self, queued_state, ctrl_trgs):
        queued_state = queued_state.replace(queued_ctrl_trgs=ctrl_trgs, has_queued_ctrl_trgs=True)
        return queued_state

    def init_graphics(self):
        self.graphics = jnp.array(self.graphics)
        # Load TTF font (Replace 'path/to/font.ttf' with the actual path)
        self.render_font = font = ImageFont.truetype("./fonts/AcPlus_IBM_VGA_9x16-2x.ttf", 20)

        ascii_chars_to_ints = {}
        self.ascii_chars_to_ims = {}

        self.render_font_shape = (16, 9)

        # Loop over a range of ASCII characters (here, printable ASCII characters from 32 to 126)
        # for i in range(0, 127):
        #     char = chr(i)

        #     # Create a blank RGBA image
        #     image = Image.new("RGBA", self.render_font_shape, (0, 0, 0, 0))

        #     # Get drawing context
        #     draw = ImageDraw.Draw(image)

        #     # Draw text
        #     draw.text((0, 0), char, font=font, fill=(255, 255, 255, 255))

        #     ascii_chars_to_ints[char] = i
        #     char_im = np.array(image)
        #     self.ascii_chars_to_ims[char] = char_im

    def observe_ctrls(self, prob_state: ProblemState):
        obs = jnp.zeros(len(self.metrics_enum))
        obs = jnp.where(self.ctrl_metrics_mask, jnp.sign(prob_state.ctrl_trgs - prob_state.stats), obs)
        # Return a vector of only the metrics we're controlling
        obs = obs[self.ctrl_metric_obs_idxs]
        return obs

    def gen_rand_ctrl_trgs(self, rng):
        # Randomly sample some control targets
        ctrl_trgs =  jnp.where(
            self.ctrl_metrics_mask,
            gen_ctrl_trgs(self.metric_bounds, rng),
            self.stat_trgs,
        )
        return ctrl_trgs

    def reset(self, env_map: chex.Array, rng, queued_state):
        ctrl_trgs = jax.lax.select(
            queued_state.has_queued_ctrl_trgs,
            queued_state.ctrl_trgs,
            self.gen_rand_ctrl_trgs(rng),
        )

        state = self.get_curr_stats(env_map)
        state = state.replace(
            ctrl_trgs=ctrl_trgs,
        )
        reward = None
        return reward, state

    def step(self, env_map: chex.Array, state: ProblemState):
        new_state = self.get_curr_stats(env_map=env_map)
        reward = get_reward(new_state.stats, state.stats, self.stat_weights, state.ctrl_trgs, self.ctrl_threshes)
        new_state = new_state.replace(
            ctrl_trgs=state.ctrl_trgs,
        )
        return reward, new_state

    def get_curr_stats(self, env_map: chex.Array) -> ProblemState:
        raise NotImplementedError

    def draw_path(self, lvl_img, env_map, border_size, path_coords_tpl, tile_size):
        assert len(path_coords_tpl) == 1
        lvl_img = draw_path(prob=self, lvl_img=lvl_img, env_map=env_map, border_size=border_size, 
                  path_coords=path_coords_tpl[0], tile_size=tile_size)
        return lvl_img


def draw_path(prob, lvl_img, env_map, border_size, path_coords, tile_size):
    # Path, if applicable
    tile_img = prob.graphics[-1]

    def draw_path_tile(carry):
        path_coords, lvl_img, i = carry
        y, x = path_coords[i]
        tile_type = env_map[y + border_size[0]][x + border_size[1]]
        empty_tile = int(Tiles.EMPTY)

        # Only draw path tiles on top of empty tiles
        lvl_img = jax.lax.cond(
            tile_type == empty_tile,
            lambda: jax.lax.dynamic_update_slice(lvl_img, tile_img,
                                            ((y + border_size[0]) * tile_size, (x + border_size[1]) * tile_size, 0)),
            lambda: lvl_img,)
                            
        return (path_coords, lvl_img, i+1)

    def cond(carry):
        path_coords, _, i = carry
        return jnp.all(path_coords[i] != jnp.array((-1, -1)))
        # return jnp.all(path_coords[i:i+env.prob.max_path_len+1] != jnp.array(-1, -1))
        # result = jnp.any(
        #     jax.lax.dynamic_slice(path_coords, (i, 0), (prob.max_path_len+1, 2)) != jnp.array((-1, -1))
        # )
        # return result

    i = 0
    _, lvl_img, _ = jax.lax.while_loop(
        cond, draw_path_tile, (path_coords, lvl_img, i))

    return lvl_img