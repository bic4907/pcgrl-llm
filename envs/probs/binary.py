from enum import IntEnum
import os
from typing import Optional, Tuple
from flax import struct
import numpy as np
from PIL import Image

from envs.pathfinding import FloodPath, FloodRegions, calc_diameter, get_max_n_regions, get_max_path_length, get_max_path_length_static, get_path_coords_diam
from envs.probs.problem import Problem, ProblemState
from envs.feature import *


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

class BinaryTiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2


@struct.dataclass
class BinaryState(ProblemState):
    flood_count: Optional[chex.Array] = None


class BinaryMetrics(IntEnum):
    DIAMETER = 0

    N_REGIONS = 1

    H_SYMMETRY = 2  # Horizontal Symmetry
    V_SYMMETRY = 3  # Vertical Symmetry
    LR_DIAGONAL_SYMMETRY = 4  # Left-to-Right Diagonal Symmetry
    RL_DIAGONAL_SYMMETRY = 5  # Right-to-Left Diagonal Symmetry

    LARGEST_COMPONENT = 6  # Largest Component Normalized

    # Local observation
    # empty spaces
    LC_R1_EMPTY_RATE = 7
    LC_R2_EMPTY_RATE = 8
    LC_R3_EMPTY_RATE = 9
    LC_R4_EMPTY_RATE = 10
    LC_R5_EMPTY_RATE = 11
    LC_R6_EMPTY_RATE = 12
    LC_R7_EMPTY_RATE = 13
    LC_R8_EMPTY_RATE = 14
    LC_R9_EMPTY_RATE = 15
    LC_R10_EMPTY_RATE = 16
    LC_R11_EMPTY_RATE = 17
    LC_R12_EMPTY_RATE = 18
    LC_R13_EMPTY_RATE = 19
    LC_R14_EMPTY_RATE = 20
    LC_R15_EMPTY_RATE = 21
    LC_R16_EMPTY_RATE = 22

    # Shannon entropy
    LC_R1_ENTROPY = 23
    LC_R2_ENTROPY = 24
    LC_R3_ENTROPY = 25
    LC_R4_ENTROPY = 26
    LC_R5_ENTROPY = 27
    LC_R6_ENTROPY = 28
    LC_R7_ENTROPY = 29
    LC_R8_ENTROPY = 30
    LC_R9_ENTROPY = 31
    LC_R10_ENTROPY = 32
    LC_R11_ENTROPY = 33
    LC_R12_ENTROPY = 34
    LC_R13_ENTROPY = 35
    LC_R14_ENTROPY = 36
    LC_R15_ENTROPY = 37
    LC_R16_ENTROPY = 38

class BinaryProblem(Problem):
    tile_enum = BinaryTiles

    tile_probs = np.zeros(len(tile_enum))
    tile_probs[BinaryTiles.EMPTY] = 0.5
    tile_probs[BinaryTiles.WALL] = 0.5
    tile_probs = tuple(tile_probs)

    tile_nums = tuple([0 for _ in range(len(tile_enum))])

    stat_weights = np.zeros(len(BinaryMetrics))
    stat_weights[BinaryMetrics.DIAMETER] = 1.0
    stat_weights[BinaryMetrics.N_REGIONS] = 1.0
    stat_weights = jnp.array(stat_weights)

    stat_trgs = np.zeros(len(BinaryMetrics))
    stat_trgs[BinaryMetrics.DIAMETER] = np.inf
    stat_trgs[BinaryMetrics.N_REGIONS] = 1
    stat_trgs = jnp.array(stat_trgs)

    metrics_enum = BinaryMetrics

    passable_tiles = jnp.array([BinaryTiles.EMPTY])

    def __init__(self, map_shape, ctrl_metrics, pinpoints):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length_static(map_shape)
        super().__init__(map_shape=map_shape, ctrl_metrics=ctrl_metrics, pinpoints=pinpoints)

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(BinaryMetrics)
        bounds[BinaryMetrics.DIAMETER] = [0, get_max_path_length(map_shape)]
        bounds[BinaryMetrics.N_REGIONS] = [0, get_max_n_regions(map_shape)]
        bounds[BinaryMetrics.H_SYMMETRY] = [0, 1]
        bounds[BinaryMetrics.V_SYMMETRY] = [0, 1]
        bounds[BinaryMetrics.LR_DIAGONAL_SYMMETRY] = [0, 1]
        bounds[BinaryMetrics.RL_DIAGONAL_SYMMETRY] = [0, 1]
        # width * height
        bounds[BinaryMetrics.LARGEST_COMPONENT] = [0, map_shape[0] * map_shape[1]]

        for idx in range(16):
            bounds[BinaryMetrics.LC_R1_EMPTY_RATE + idx] = [0, 1]
            bounds[BinaryMetrics.LC_R1_ENTROPY + idx] = [0, 1]



        return jnp.array(bounds)

    def get_path_coords(self, env_map: chex.Array, prob_state: BinaryState) -> Tuple[chex.Array]:
        return (get_path_coords_diam(flood_count=prob_state.flood_count, max_path_len=self.max_path_len),)
    
    def init_graphics(self):
        self.graphics = [0] * (len(self.tile_enum) + 1)
        self.graphics[BinaryTiles.EMPTY] = Image.open(
                f"{__location__}/tile_ims/empty.png"
            ).convert('RGBA')
        self.graphics[BinaryTiles.WALL] = Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA')
        self.graphics[BinaryTiles.BORDER] = Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA')
        self.graphics[len(self.tile_enum)] = Image.open(f"{__location__}/tile_ims/path_g.png").convert(
                'RGBA')

        super().init_graphics()

    def get_curr_stats(self, env_map: chex.Array):
        """Get relevant metrics from the current state of the environment."""
        diameter, flood_path_state, n_regions, flood_regions_state = calc_diameter(
            self.flood_regions_net, self.flood_path_net, env_map, self.passable_tiles
        )

        # Initialize stats array with zeros
        stats = jnp.zeros(len(BinaryMetrics))

        # Set diameter and number of regions
        stats = stats.at[BinaryMetrics.DIAMETER].set(diameter)
        stats = stats.at[BinaryMetrics.N_REGIONS].set(n_regions)

        # Calculate the largest component of WALL tiles
        stats = stats.at[BinaryMetrics.LARGEST_COMPONENT].set(get_largest_component_size(flood_regions_state.flood_count))

        # Calculate symmetry metrics (consider only 1 and 2 blocks)
        h_symmetry = calculate_horizontal_symmetry(env_map)
        v_symmetry = calculate_vertical_symmetry(env_map)
        lr_diagonal_symmetry = calculate_lr_diagonal_symmetry(env_map)
        rl_diagonal_symmetry = calculate_rl_diagonal_symmetry(env_map)

        stats = stats.at[BinaryMetrics.H_SYMMETRY].set(h_symmetry)
        stats = stats.at[BinaryMetrics.V_SYMMETRY].set(v_symmetry)
        stats = stats.at[BinaryMetrics.LR_DIAGONAL_SYMMETRY].set(lr_diagonal_symmetry)
        stats = stats.at[BinaryMetrics.RL_DIAGONAL_SYMMETRY].set(rl_diagonal_symmetry)

        empty_block_rates = self.calculate_empty_block_rates(env_map)
        for idx in range(16):
            stats = stats.at[BinaryMetrics.LC_R1_EMPTY_RATE + idx].set(empty_block_rates[idx])

        # Calculate Shannon entropy for each region
        entropies = calculate_entropy_for_regions(env_map, num_unique_values=self.get_num_unique_tiles())
        for idx in range(16):
            stats = stats.at[BinaryMetrics.LC_R1_ENTROPY + idx].set(entropies[idx])

        # Return state with the new stats
        state = BinaryState(
            stats=stats, flood_count=flood_path_state.flood_count, ctrl_trgs=None
        )
        return state

    def calculate_empty_block_rates(self, env_map: chex.Array) -> chex.Array:
        """
        Calculate the rate of empty blocks in each region.
        We divide the map into 16 regions (4x4) and calculate the empty block rate for each.
        """
        map_height, map_width = env_map.shape
        region_height, region_width = map_height // 4, map_width // 4

        empty_rates = jnp.zeros(16)

        for i in range(4):
            for j in range(4):
                region = env_map[i * region_height:(i + 1) * region_height, j * region_width:(j + 1) * region_width]
                n_empty_blocks = jnp.sum(region == BinaryTiles.EMPTY)
                total_blocks = region_height * region_width
                region_idx = i * 4 + j
                empty_rates = empty_rates.at[region_idx].set(n_empty_blocks / total_blocks)

        return empty_rates

    def get_num_unique_tiles(self):
        return len(BinaryTiles) - 1  # Subtract 1 to exclude BORDER