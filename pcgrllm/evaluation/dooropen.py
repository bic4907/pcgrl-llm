import logging
import os
import shutil
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from os.path import dirname, join, basename
from typing import Tuple

from flax import struct

from envs.pathfinding import calc_path_from_a_to_b, FloodPath, calc_path_length
from envs.probs.dungeon2 import Dungeon2Tiles
from envs.probs.dungeon3 import Dungeon3Tiles, Dungeon3Problem
from envs.solution import get_solution_jit, get_min_distance_jit
from pcgrllm.evaluation.base import LevelEvaluator, EvaluationResult
from pcgrllm.scenario_preset import ScenarioPreset
from pcgrllm.task import TaskType
from pcgrllm.utils.cuda import get_cuda_version
from pcgrllm.utils.storage import Iteration


"""
# 시나리오 계획
- 키를 주워서 문을 열고, 들어가서 문안에 있는 treasure을 만날 수 있을지?
- BAT이 키를 가지고 있어서 죽여서 키를 얻어야함. 문을 열고 Treasure을 만날 수 있을지?

=> 기능상 [KEY, BAT] -> [DOOR] -> [TREASURE]이 연결되어야 하고
=> [KEY, BAT] <=> [TREASURE]은 연결되어야하지 않음
"""

# NOT_EXISTS = jnp.array([-1, -1])
#
# ENUM_TO_ENCOUNTER_INDEX = {
#     Dungeon3Tiles.BAT: 0,
#     Dungeon3Tiles.SCORPION: 1,
#     Dungeon3Tiles.SPIDER: 2,
# }
#
# tile_keys = jnp.array(list(ENUM_TO_ENCOUNTER_INDEX.keys()))
# tile_indices = jnp.array(list(ENUM_TO_ENCOUNTER_INDEX.values()))

CUDA_VERSION = get_cuda_version()


@struct.dataclass
class EvaluationResultStruct:
    playability: float = 0
    naive_playability: float = 0
    solvability: float = 0
    is_closed: float = 0
    imp_reachable: float = 0

def remove_item_from_array(array, item):
    mask = array != item
    idx = jnp.where(mask, size=array.shape[0], fill_value=-1)[0]
    return array[idx][:-1]

def eval_level(level: np.ndarray, scenario_num) -> Tuple[float, float]:
    '''
    # Scenario generation task
    '''

    if len(level.shape) != 2:
        raise ValueError(f"The level must be 2D array. Got {len(level.shape)}D array.")

    passable_tiles = Dungeon3Problem.passable_tiles
    passable_tiles_wo_spider = remove_item_from_array(passable_tiles, Dungeon3Tiles.SPIDER)

    imp_tiles = ScenarioPreset().scenarios[str(scenario_num)].important_tiles
    imp_tile = jnp.array(imp_tiles)[0]

    p_xy = jnp.argwhere(level == Dungeon3Tiles.PLAYER, size=1, fill_value=-1)[0]
    d_xy = jnp.argwhere(level == Dungeon3Tiles.DOOR, size=1, fill_value=-1)[0]

    p_d_length, _, _ = calc_path_from_a_to_b(level, passable_tiles_wo_spider, p_xy, d_xy)
    is_closed = jnp.where(p_d_length < 0, 1, 0)  # 길이 0보다 크면 연결되어있음


    #
    p_d_length = get_min_distance_jit(env_map=level,
                                    source_tile=Dungeon3Tiles.PLAYER,
                                    target_tile=Dungeon3Tiles.SPIDER, # door
                                    passable_tiles=passable_tiles)
    is_p_d_reachable = jnp.where(p_d_length > 0, 1, 0)
    #
    p_i_dist = get_min_distance_jit(env_map=level,
                                    source_tile=Dungeon3Tiles.PLAYER,
                                    target_tile=imp_tile,
                                    passable_tiles=passable_tiles)
    is_p_i_reachable = jnp.where(p_i_dist > 0, 1, 0)


    t_d_length = get_min_distance_jit(env_map=level,
                                    source_tile=Dungeon3Tiles.DOOR, # treasure
                                    target_tile=Dungeon3Tiles.SPIDER, # door
                                    passable_tiles=passable_tiles)
    is_t_d_reachable = jnp.where(t_d_length > 0, 1, 0)

    is_solvable =(is_p_d_reachable == 1) & (is_p_i_reachable == 1)
    is_naive_playable = (is_solvable == 1) & (is_t_d_reachable == 1)
    is_playable = (is_naive_playable == 1) & (is_closed == 1)

    return EvaluationResultStruct(
        playability=is_playable, # 열쇠를 주워서 문을 열 수 있고, 문안에 있는 treasure을 만날 수 있는지
        naive_playability=is_naive_playable, # 플레이어가 문까지 도달할 수 있는지
        solvability=is_solvable, # 열쇠를 주워서 문을 열 수 있는데 까지
        is_closed=is_closed,
        imp_reachable=is_p_i_reachable,
    )


def eval_level_jax(levels, scenario_num):

    def eval_level_wrapper(level):
        return eval_level(level, scenario_num=scenario_num)

    # CUDA 버전에 따라 병렬 처리 방식 선택
    if CUDA_VERSION is None or CUDA_VERSION > 12:
        results = jax.vmap(eval_level_wrapper)(levels)
    else:
        results = jax.lax.map(eval_level_wrapper, levels)

    # return (results.playability, results.solvability, results.is_closed, results.imp_reachable)
    return (results.playability, results.naive_playability, results.solvability,
            results.is_closed, results.imp_reachable)

class DooropenEvaluator(LevelEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, iteration: Iteration, scenario_num: str = None, target_character: str = None, visualize: bool = False, use_train: bool = False, step_filter=None) -> EvaluationResult:

        numpy_files = iteration.get_numpy_files(train=use_train, step_filter=step_filter)

        scenario_num = scenario_num if scenario_num is not None else target_character
        levels = jnp.array([numpy_file.load() for numpy_file in numpy_files])

        eval_results = eval_level_jax(levels=levels, scenario_num=scenario_num)


        (playability, naive_playability, solvability, is_closed, imp_reachable) = eval_results

        # 평균 계산
        playability = jnp.mean(playability)
        naive_playability = jnp.mean(naive_playability)
        solvability = jnp.mean(solvability)
        is_closed = jnp.mean(is_closed)
        imp_reachable = jnp.mean(imp_reachable)
        sample_size = len(levels)

        return EvaluationResult(
            task=self.task,
            playability=playability,
            naive_playability=naive_playability,
            solvability=solvability,
            is_closed=is_closed,
            acc_imp_perc=imp_reachable,
            sample_size=sample_size,
        )

if __name__ == '__main__':
    # Initialize logger
    from debug.scenario_levels import DooropenLevels

    logger = logging.getLogger(basename(__file__))
    logger.setLevel(logging.DEBUG)

    evaluator = DooropenEvaluator(logger=logger, task=TaskType.Scenario2)

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
    for idx, level in enumerate(DooropenLevels[:]):
        np.save(join(iteration.get_numpy_dir(), f"level_{idx}.npy"), level)
    # Run the evaluator with visualization enabled/disabled
    result = evaluator.run(iteration=iteration, scenario_num="5", visualize=True)
    print(result)

    result = evaluator.run(iteration=iteration, scenario_num="6", visualize=True)
    print(result)
