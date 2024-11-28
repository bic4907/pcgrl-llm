from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from os.path import dirname, join, basename
from typing import Tuple

from jax import jit

from envs.pathfinding import calc_path_from_a_to_b, check_event
from envs.probs.dungeon3 import Dungeon3Tiles, Dungeon3Problem
from pcgrllm.evaluation.base import *
from pcgrllm.scenario_preset import ScenarioPreset
from pcgrllm.utils.storage import Iteration

NOT_EXISTS = jnp.array([-1, -1])

TILE_MAP_STR_ENUM = {
    "BORDER": Dungeon3Tiles.BORDER,
    "EMPTY": Dungeon3Tiles.EMPTY,
    "WALL": Dungeon3Tiles.WALL,
    "PLAYER": Dungeon3Tiles.PLAYER,
    "BAT": Dungeon3Tiles.BAT,
    "SCORPION": Dungeon3Tiles.SCORPION,
    "SPIDER": Dungeon3Tiles.SPIDER,
    "KEY": Dungeon3Tiles.KEY,
    "DOOR": Dungeon3Tiles.DOOR,
}


partial(jit, static_argnums=(0, 1))
def eval_level(level: np.ndarray, scenario_num) -> Tuple[float, float]:
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
    # if level shape is not 2d, raise error
    if len(level.shape) != 2:
        raise ValueError(f"The level must be 2D array. Got {len(level.shape)}D array.")
    #
    passable_tiles = Dungeon3Problem.passable_tiles
    imp_tiles = ScenarioPreset().scenarios[str(scenario_num)].important_tiles

    p_xy = jnp.argwhere(level == Dungeon3Tiles.PLAYER, size=1, fill_value=-1)[0]
    d_xy = jnp.argwhere(level == Dungeon3Tiles.DOOR, size=1, fill_value=-1)[0]
    k_xy = jnp.argwhere(level == Dungeon3Tiles.KEY, size=1, fill_value=-1)[0]
    #
    p_t_length, _, _ = calc_path_from_a_to_b(level, passable_tiles, p_xy, d_xy)
    # p_t_length = 0
    # p_t_connected = 0
    p_t_connected = jnp.where(p_t_length >= 0, 1, 0)

    if (k_xy != NOT_EXISTS).all():
        p_k_length, _, _ = calc_path_from_a_to_b(level, passable_tiles, p_xy, k_xy)
    else:
        p_k_length = -1

    p_k_connected = jnp.where(p_k_length > 0, 1, 0)
    is_solavable = jnp.where(jnp.logical_and(p_t_connected, p_k_connected), 1, 0)

    n_exist_imp_tiles = 0
    n_acc_imp_tiles = 0

    for imp_tile in imp_tiles:
        tile_num = TILE_MAP_STR_ENUM[imp_tile]

        _xy = jnp.argwhere(level == tile_num, size=1, fill_value=-1)[0]
        if (_xy != NOT_EXISTS).all():
            n_exist_imp_tiles += 1

            _dist, _, _ = calc_path_from_a_to_b(level, passable_tiles, p_xy, _xy)

            if _dist > 0:
                n_acc_imp_tiles += 1

    n_solutions = 0
    for key in jnp.argwhere(level == Dungeon3Tiles.KEY):
        cnt, solutions = check_event(env_map=level,
                                     passable_tiles=passable_tiles,
                                     src=p_xy,
                                     key=key,
                                     trg=d_xy)
        n_solutions += cnt

    # check if the player and door is connected
    playability = p_t_connected
    path_length = p_t_length
    solvability = is_solavable
    n_solutions = n_solutions
    loss_solutions = len(imp_tiles) - n_solutions
    acc_imp_tiles = n_acc_imp_tiles / len(imp_tiles)
    exist_imp_tiles = n_exist_imp_tiles / len(imp_tiles)


    return EvaluationResult(
        task=TaskType.Scenario,
        playability=playability,
        path_length=path_length,
        solvability=solvability,
        n_solutions=n_solutions,
        loss_solutions=loss_solutions,
        acc_imp_perc=acc_imp_tiles,
        exist_imp_perc=exist_imp_tiles,
        sample_size=1)



class SolutionEvaluator(LevelEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run(self, iteration: Iteration, scenario_num: str, visualize: bool = False, use_train: bool = False, step_filter=None) -> EvaluationResult:
        numpy_files = iteration.get_numpy_files(train=use_train, step_filter=step_filter)

        results = []
        for numpy_file in numpy_files:
            level = numpy_file.load()

            result = eval_level(level, scenario_num=scenario_num)
            results.append(result)


        # Calculate the average of the results
        playability = np.mean([result.playability for result in results])
        path_length = np.mean([result.path_length for result in results])
        solvability = np.mean([result.solvability for result in results])
        n_solutions = np.mean([result.n_solutions for result in results])
        loss_solutions = np.mean([result.loss_solutions for result in results])
        acc_imp_tiles = np.mean([result.acc_imp_perc for result in results])
        exist_imp_tiles = np.mean([result.exist_imp_perc for result in results])
        sample_size = len(results)

        return EvaluationResult(
            task=self.task,
            playability=playability,
            path_length=path_length,
            solvability=solvability,
            n_solutions=n_solutions,
            loss_solutions=loss_solutions,
            acc_imp_tiles=acc_imp_tiles,
            exist_imp_tiles=exist_imp_tiles,
            sample_size=sample_size)

# Example


if __name__ == '__main__':
    # Initialize logger
    from debug.scenario_levels import AllLevels


    logger = logging.getLogger(basename(__file__))
    logger.setLevel(logging.DEBUG)

    evaluator = SolutionEvaluator(logger=logger, task=TaskType.Scenario)


    base_path = join(dirname(__file__), 'example')
    # Define the path for the iteration folder
    example_path = join(base_path, 'scenario_1', 'iteration_1')

    # Load the iteration
    iteration = Iteration.from_path(path=example_path)

    # remove numpy files in the directory
    # import os
    # os.system(f"rm -rf {iteration.get_numpy_dir()}/*")

    # save the alllevels into the numpy dir
    for idx, level in enumerate(AllLevels[:]):
        np.save(join(iteration.get_numpy_dir(), f"level_{idx}.npy"), level)

    # Run the evaluator with visualization enabled/disabled
    result = evaluator.run(iteration=iteration, scenario_num="1", visualize=True)
