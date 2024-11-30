import os
import shutil
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from os.path import dirname, join, basename
from typing import Tuple

from jax import jit

from envs.pathfinding import calc_path_from_a_to_b, check_event, erase_unnecessary_arr
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

    p_t_length, _, _ = calc_path_from_a_to_b(level, passable_tiles, p_xy, d_xy)
    p_t_connected = jnp.where(p_t_length >= 0, 1, 0)

    p_k_length = jax.lax.cond(
        jnp.all(k_xy != NOT_EXISTS),
        lambda _: calc_path_from_a_to_b(level, passable_tiles, p_xy, k_xy)[0],
        lambda _: jnp.array(-1).astype(float),
        operand=None
    )

    p_k_connected = jnp.where(p_k_length > 0, 1, 0)
    is_solavable = jnp.where(jnp.logical_and(p_t_connected, p_k_connected), 1, 0)

    def process_important_tile(time_num):
        _xy = jnp.argwhere(level == time_num, size=1, fill_value=-1)[0]

        def tile_exists(_xy):
            _passable_tiles = jnp.append(Dungeon3Problem.passable_tiles, time_num)
            _dist, _, _ = calc_path_from_a_to_b(level, _passable_tiles, p_xy, _xy)
            n_acc = jnp.where(_dist > 0, 1, 0)
            return 1, n_acc  # Tile exists and may be reachable

        def tile_not_exists(_xy):
            return 0, 0  # Tile does not exist

        return jax.lax.cond(
            (_xy != NOT_EXISTS).all(),
            tile_exists,
            tile_not_exists,
            _xy
        )

    # Use vmap to process all important tiles in parallel
    imp_tiles = jnp.array([TILE_MAP_STR_ENUM[tile] for tile in imp_tiles])
    imp_tile_results = jax.vmap(process_important_tile)(jnp.array(imp_tiles))

    # Summing results
    n_exist_imp_tiles = jnp.sum(imp_tile_results[0])  # Count of existing important tiles
    n_acc_imp_tiles = jnp.sum(imp_tile_results[1])   # Count of reachable important tiles


    n_solutions = 0
    encounter_monster = {}
    exist_keys = jnp.argwhere(level == Dungeon3Tiles.KEY, size=30, fill_value=-1)
    exist_keys = erase_unnecessary_arr(exist_keys)
    for key in exist_keys:
        encounter_monster_num, route = check_event(env_map=level,
                                                       passable_tiles=passable_tiles,
                                                       src=p_xy,
                                                       key=key,
                                                       trg=d_xy,
                                                       exist_keys=exist_keys)
        if route:
            n_solutions += 1
            encounter_monster = {**encounter_monster, **encounter_monster_num}
            print(encounter_monster)

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

        # 로드한 numpy 파일을 JAX 배열로 변환
        levels = jnp.array([numpy_file.load() for numpy_file in numpy_files])

        # JAX에서 사용 가능한 eval_level 함수로 작성되어야 함
        def eval_level_jax(level):
            # eval_level 함수가 JAX와 호환되도록 구현되어야 함
            result = eval_level(level, scenario_num=scenario_num)
            return (
                result.playability,
                result.path_length,
                result.solvability,
                result.n_solutions,
                result.loss_solutions,
                result.acc_imp_perc,
                result.exist_imp_perc,
            )

        # 병렬화를 적용
        # eval_results = jax.vmap(eval_level_jax)(levels)
        eval_results = eval_level_jax(levels[0])


        # 결과를 개별적으로 계산
        playability, path_length, solvability, n_solutions, loss_solutions, acc_imp_tiles, exist_imp_tiles = eval_results

        # 평균 계산
        playability = jnp.mean(playability)
        path_length = jnp.nan_to_num(
            jnp.nanmean(jnp.where(path_length != -1, path_length, jnp.nan))
        )
        solvability = jnp.mean(solvability)
        n_solutions = jnp.mean(n_solutions)
        loss_solutions = jnp.mean(loss_solutions)
        acc_imp_tiles = jnp.mean(acc_imp_tiles)
        exist_imp_tiles = jnp.mean(exist_imp_tiles)

        sample_size = len(levels)

        return EvaluationResult(
            task=self.task,
            playability=playability,
            path_length=path_length,
            solvability=solvability,
            n_solutions=n_solutions,
            loss_solutions=loss_solutions,
            acc_imp_perc=acc_imp_tiles,
            exist_imp_perc=exist_imp_tiles,
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

    numpy_dir = iteration.get_numpy_dir()

    if os.path.exists(numpy_dir):
        shutil.rmtree(numpy_dir)  # 디렉토리 자체를 삭제
        os.makedirs(numpy_dir)  # 빈 디렉토리 다시 생성

    # save the alllevels into the numpy dir
    for idx, level in enumerate(AllLevels[:]):
        np.save(join(iteration.get_numpy_dir(), f"level_{idx}.npy"), level)
    # Run the evaluator with visualization enabled/disabled
    result = evaluator.run(iteration=iteration, scenario_num="1", visualize=True)
    print(result)
