import random
from enum import IntEnum
import math
import os
from functools import partial

import chex
from flax import struct
import jax
import jax.numpy as jnp
from PIL import Image
import numpy as np

from envs.pathfinding import FloodPath, FloodPathState, FloodRegions, FloodRegionsState, calc_diameter, calc_n_regions, calc_path_length, get_max_n_regions, get_max_path_length, get_max_path_length_static, get_path_coords
from envs.probs.dungeon2 import Dungeon2Metrics
from envs.probs.problem import Problem, ProblemState, draw_path, get_max_loss, get_reward, gen_init_map, MapData
from envs.utils import idx_dict_to_arr, Tiles


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Dungeon3Tiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2
    PLAYER = 3
    BAT = 4
    SCORPION = 5
    SPIDER = 6
    KEY = 7
    DOOR = 8


class Dungeon3Metrics(IntEnum):
    N_REGIONS = 0

@struct.dataclass
class Dungeon3State(ProblemState):
    pass

class Dungeon3Problem(Problem):
    tile_enum = Dungeon3Tiles
    metrics_enum = Dungeon3Metrics

    ctrl_threshes = np.zeros(len(Dungeon3Metrics))
    ctrl_threshes[Dungeon3Metrics.N_REGIONS] = 2

    ctrl_metrics_mask = np.zeros(len(Dungeon3Metrics))

    randomize_start_pos: bool = False
    unavailable_tiles: list = [Dungeon3Tiles.PLAYER, Dungeon3Tiles.DOOR]

    tile_probs = {
        Dungeon3Tiles.BORDER: 0.0,
        Dungeon3Tiles.EMPTY: 0.58,
        Dungeon3Tiles.WALL: 0.5,
        Dungeon3Tiles.PLAYER: 0.00, # Don't place this block on initialization
        Dungeon3Tiles.KEY: 0.005,
        Dungeon3Tiles.DOOR: 0.00,  # Don't place this block on initialization
        Dungeon3Tiles.BAT: 0.01,
        Dungeon3Tiles.SCORPION: 0.01,
        Dungeon3Tiles.SPIDER: 0.01,
    }
    tile_probs = tuple(idx_dict_to_arr(tile_probs))

    stat_weights = {
        Dungeon3Metrics.N_REGIONS: 1,
    }
    stat_weights = idx_dict_to_arr(stat_weights)

    tile_nums = [0 for _ in range(len(tile_enum))]
    tile_nums[Dungeon3Tiles.PLAYER] = 1
    tile_nums[Dungeon3Tiles.DOOR] = 1
    tile_nums[Dungeon3Tiles.KEY] = 1
    tile_nums = tuple(tile_nums)

    # Passible tiles 등록하기
    passable_tiles = jnp.array([Dungeon3Tiles.EMPTY, Dungeon3Tiles.KEY,])

    def __init__(self, map_shape, ctrl_metrics, pinpoints, randomize_start_pos):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length_static(map_shape)
        self.n_tiles = math.prod(map_shape)
        self.randomize_start_pos = randomize_start_pos

        stat_trgs = {
            Dungeon3Metrics.N_REGIONS: 1,
        }
        self.stat_trgs = idx_dict_to_arr(stat_trgs)
        self.ctrl_threshes[Dungeon2Metrics.N_REGIONS] = 1

        super().__init__(map_shape=map_shape, ctrl_metrics=ctrl_metrics, pinpoints=pinpoints)


    @partial(jax.jit, static_argnames=("self", "randomize_map_shape", "empty_start", "pinpoints"))
    def gen_init_map(self, rng, randomize_map_shape=False, empty_start=False, pinpoints=False):
        init_map = gen_init_map(rng, self.tile_enum, self.map_shape, self.tile_probs,
                                randomize_map_shape=randomize_map_shape,
                                empty_start=empty_start,
                                tile_nums=self.tile_nums,
                                pinpoints=pinpoints)

        org_map = init_map.env_map
        map_size_x, map_size_y = org_map.shape  # 맵의 크기 추출

        # `static_map` 초기화 (모두 False로 시작)
        static_map = jnp.zeros((map_size_x, map_size_y), dtype=bool)

        def padded_edge_positions():
            # 패딩 1을 적용한 가장자리에서 4개의 후보군 생성
            candidates = jnp.array([
                [1, 1],  # 좌상단 (1,1)
                [1, map_size_y - 2],  # 우상단 (1, width-2)
                [map_size_x - 2, map_size_y - 2],  # 우하단 (height-2, width-2)
                [map_size_x - 2, 1],  # 좌하단 (height-2, 1)
            ])
            return candidates

        def diagonal_positions(key):
            # 대각선 위치를 계산
            candidates = padded_edge_positions()
            idx = jax.random.choice(key, jnp.arange(4), (1,))[0]
            player_pos = candidates[idx]
            # 대각선 반대 위치
            door_pos = candidates[(idx + 2) % 4]
            return player_pos, door_pos

        player_pos, door_pos = diagonal_positions(rng)

        # 플레이어와 문 배치
        org_map = org_map.at[player_pos[0], player_pos[1]].set(Dungeon3Tiles.PLAYER)
        org_map = org_map.at[door_pos[0], door_pos[1]].set(Dungeon3Tiles.DOOR)

        # `static_map`에 플레이어와 문 위치를 True로 설정
        static_map = static_map.at[player_pos[0], player_pos[1]].set(True)
        static_map = static_map.at[door_pos[0], door_pos[1]].set(True)

        # 주변을 EMPTY SPACE로 채우기 위한 보조 함수
        def set_empty_around(pos, map_array, static_array):
            x, y = pos
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    map_array, static_array = jax.lax.cond(
                        (0 <= nx) & (nx < map_size_x) & (0 <= ny) & (ny < map_size_y) & (
                                map_array[nx, ny] != Dungeon3Tiles.PLAYER) & (
                                map_array[nx, ny] != Dungeon3Tiles.DOOR),
                        lambda arrays: (
                            arrays[0].at[nx, ny].set(Dungeon3Tiles.EMPTY),
                            arrays[1].at[nx, ny].set(True)
                        ),
                        lambda arrays: arrays,
                        (map_array, static_array),
                    )
            return map_array, static_array

        # 플레이어와 문 주변 EMPTY SPACE로 채우기
        org_map, static_map = set_empty_around(player_pos, org_map, static_map)
        org_map, static_map = set_empty_around(door_pos, org_map, static_map)

        # 새 맵 데이터 반환
        new_init_map = MapData(org_map, init_map.actual_map_shape, static_map=static_map)

        return new_init_map

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(Dungeon3Metrics)
        bounds[Dungeon3Metrics.N_REGIONS] = [0, self.max_path_len * 2]
        bounds = jnp.array(bounds, dtype=jnp.float32)
        return bounds

    def get_path_coords(self, env_map: chex.Array, prob_state: Dungeon3State):
        return tuple()

    def get_curr_stats(self, env_map: chex.Array):
        stats = jnp.zeros(len(Dungeon3Metrics))
        stats = stats.at[Dungeon3Metrics.N_REGIONS].set(0)

        state = Dungeon3State(
            stats=stats,
            ctrl_trgs=None,
        )
        return state

    def init_graphics(self):

        self.graphics = {
            Dungeon3Tiles.EMPTY: Image.open(
                f"{__location__}/tile_ims/empty.png"
            ).convert('RGBA'),
            Dungeon3Tiles.WALL: Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA'),
            Dungeon3Tiles.BORDER: Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA'),
            Dungeon3Tiles.KEY: Image.open(
                f"{__location__}/tile_ims/key.png"
            ).convert('RGBA'),
            Dungeon3Tiles.DOOR: Image.open(
                f"{__location__}/tile_ims/door.png"
            ).convert('RGBA'),
            Dungeon3Tiles.PLAYER: Image.open(
                f"{__location__}/tile_ims/player.png"
            ).convert('RGBA'),
            Dungeon3Tiles.BAT: Image.open(
                f"{__location__}/tile_ims/bat.png"
            ).convert('RGBA'),
            Dungeon3Tiles.SCORPION: Image.open(
                f"{__location__}/tile_ims/scorpion.png"
            ).convert('RGBA'),
            Dungeon3Tiles.SPIDER: Image.open(
                f"{__location__}/tile_ims/spider.png"
            ).convert('RGBA'),
            len(Dungeon3Tiles): Image.open(f"{__location__}/tile_ims/path_g.png").convert(
                'RGBA'
            ),
            len(Dungeon3Tiles) + 1: Image.open(f"{__location__}/tile_ims/path_purple.png").convert(
                'RGBA'
            )
        }
        self.graphics = jnp.array(idx_dict_to_arr(self.graphics))
        super().init_graphics()

    def draw_path(self, lvl_img,env_map, border_size, path_coords_tpl, tile_size):
        return lvl_img


