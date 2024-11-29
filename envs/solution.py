from flax import struct

import chex
import jax.numpy as jnp
from gradio.themes.builder_app import palette_range
from seaborn import color_palette

from debug.render_level import generate_color_palette
from envs.pathfinding import check_event
from envs.probs.dungeon3 import Dungeon3Tiles, Dungeon3Problem


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

    sol_cnt = 0
    for i, key in enumerate(jnp.argwhere(env_map == Dungeon3Tiles.KEY)):
        _cnt, _solutions = check_event(env_map=env_map,
                                     passable_tiles=passable_tiles,
                                     src=p_xy,
                                     key=key,
                                     trg=d_xy)

        for path in _solutions:
            color = color_palette[sol_cnt % len(color_palette)]
            offset = jnp.array([0, 0])
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


