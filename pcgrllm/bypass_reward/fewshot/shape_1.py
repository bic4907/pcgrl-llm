import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from debug.render_level import render_level
from envs.pathfinding import calc_path_from_a_to_b

@jit
def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    reward = 0.0

    EMPTY = 1
    WALL = 2
    PLAYER = 3
    BAT = 4
    SCORPION = 5
    SPIDER = 6
    KEY = 7
    DOOR = 8


    def tile_loss(prev_array, curr_array, tile, tgt_count):
        # Get the number of keys using jnp
        prev_cnt = jnp.sum(prev_array == tile)
        curr_cnt = jnp.sum(curr_array == tile)

        prev_loss = abs(prev_cnt - tgt_count)
        curr_loss = abs(curr_cnt - tgt_count)
        key_loss = prev_loss - curr_loss

        return key_loss

    passible_tiles = jnp.array([EMPTY, DOOR, PLAYER])

    xy_player = jnp.argwhere(curr_array == PLAYER, size=1)
    xy_door = jnp.argwhere(curr_array == DOOR, size=1)


    key_loss = tile_loss(prev_array, curr_array, KEY, 1)
    reward += key_loss * 1.0

    prev_p_d_length, _, _ = calc_path_from_a_to_b(prev_array, passible_tiles, xy_player[0], xy_door[0])
    curr_p_d_length, _, _ = calc_path_from_a_to_b(curr_array, passible_tiles, xy_player[0], xy_door[0])

    is_better = jnp.bitwise_and(prev_p_d_length == -1, curr_p_d_length >= 0)
    reward = jnp.where(is_better, reward + 3.0, reward)

    bat_loss = tile_loss(prev_array, curr_array, BAT, 1)
    scol_loss = tile_loss(prev_array, curr_array, SCORPION, 0)
    spi_loss = tile_loss(prev_array, curr_array, SPIDER, 0)

    reward += bat_loss * 1.0 + scol_loss * 0.5 + spi_loss * 0.5

    return reward

