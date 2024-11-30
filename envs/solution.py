import warnings

from flax import struct
import jax
import chex
import jax.numpy as jnp

from envs.pathfinding import check_event
from envs.probs.dungeon3 import Dungeon3Tiles, Dungeon3Problem
from envs.utils import generate_color_palette, generate_offset_palette


@struct.dataclass
class Solution:
    index: int

    color: chex.Array
    path: chex.Array
    offset: chex.Array
    layover: chex.Array

@struct.dataclass
class Solutions:
    n: int

    solutions: chex.Array


def get_solution(env_map) -> Solutions:
    '''
      # Scenario generation task
      playability: float = 0  # If the player can reach to the door
      path_length: float = 0  # The path length from the player to the door
      solvability: float = 0  # if the player can reach the door with the key
      n_solutions: float = 0  # Number of solutions the player can reach the door
      loss_solutions: float = 0  # Loss of between the target and the current solution count
      acc_imp_tiles: float = 0  # (reachable of important tiles <-> prompt)
      exist_imp_tiles: float = 0  # (existence of important tiles <-> prompt)
      '''

    p_xy = jnp.argwhere(env_map == Dungeon3Tiles.PLAYER, size=1, fill_value=-1)[0]
    d_xy = jnp.argwhere(env_map == Dungeon3Tiles.DOOR, size=1, fill_value=-1)[0]

    passable_tiles = Dungeon3Problem.passable_tiles

    solutions = list()
    color_palette = generate_color_palette(30)
    offset_palette = generate_offset_palette(9)

    # TODO Change the size to N
    sol_cnt = 0
    exist_keys = jnp.argwhere(env_map == Dungeon3Tiles.KEY, size=30, fill_value=-1)
    exist_keys = exist_keys[jnp.all(exist_keys != jnp.array([-1, -1]), axis=1)]
    print(exist_keys)
    for i, key in enumerate(exist_keys):
        _cnt, _solutions = check_event(env_map=env_map,
                                     passable_tiles=passable_tiles,
                                     src=p_xy,
                                     key=key,
                                     trg=d_xy,
                                        exist_keys=exist_keys)
        if _solutions:
            # for solution in _solutions:
            path = jnp.concatenate(_solutions)
            color = color_palette[sol_cnt % len(color_palette)]
            offset = offset_palette[sol_cnt % len(offset_palette)]

            solution = Solution(index=i, color=color, path=path, offset=offset, layover=key)
            solutions.append(solution)
            sol_cnt += 1

    solutions = Solutions(n=sol_cnt, solutions=solutions)

    return solutions



if __name__ == '__main__':
    from debug.scenario_levels import AllLevels

    level = AllLevels[1]

    solutions = get_solution(level)
    print(solutions)

    try:
        # with jax
        jit = jax.jit(get_solution)
        solutions = jit(level)
    except:
        # without jax
        warnings.warn('JIT is not available')


