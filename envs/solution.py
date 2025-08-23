import warnings
from functools import partial

from flax import struct
import jax
import chex
import jax.numpy as jnp
from jax import jit

from envs.pathfinding import check_event, check_event_jit, get_max_path_length_static, calc_path_from_a_to_b
from envs.probs.dungeon3 import Dungeon3Tiles, Dungeon3Problem
from envs.utils import generate_color_palette, generate_offset_palette

@struct.dataclass
class Solutions:
    n: chex.Array
    index: chex.Array
    color: chex.Array
    path: chex.Array
    offset: chex.Array
    layover: chex.Array
    enemy_encounter: chex.Array



def get_solution(env_map) -> Solutions:
    """
    Generate solutions for a scenario and store them in a Solutions object.
    """

    p_xy = jnp.argwhere(env_map == Dungeon3Tiles.PLAYER, size=1, fill_value=-1)[0]
    d_xy = jnp.argwhere(env_map == Dungeon3Tiles.DOOR, size=1, fill_value=-1)[0]

    passable_tiles = Dungeon3Problem.passable_tiles

    color_palette = generate_color_palette(30)
    offset_palette = generate_offset_palette(9)

    exist_keys = jnp.argwhere(env_map == Dungeon3Tiles.KEY, size=30, fill_value=-1)

    # Initialize maximum solution size (e.g., 30)
    max_solutions = 30

    max_path_len = get_max_path_length_static(map_shape=env_map.shape)


    # Initialize arrays for solutions
    indices = jnp.full((max_solutions,), -1, dtype=jnp.int32)
    colors = jnp.zeros((max_solutions, 4), dtype=jnp.uint8)  # Assuming RGBA colors
    paths = jnp.full((max_solutions, max_path_len * 2, 2), -1, dtype=jnp.int32)
    offsets = jnp.zeros((max_solutions, 2), dtype=jnp.int32)
    layovers = jnp.full((max_solutions, 2), -1, dtype=jnp.int32)
    enemy_encounters = jnp.zeros((max_solutions, 3), dtype=jnp.int32)

    sol_cnt = 0

    # Loop over keys and compute solutions
    for i, key in enumerate(exist_keys):

        # Skip invalid keys
        if jnp.all(key == jnp.array([-1, -1])):
            continue

        # Check for valid solutions
        has_solution, enemy_encounter, path = check_event_jit(
            env_map=env_map,
            passable_tiles=passable_tiles,
            src=p_xy,
            key=key,
            trg=d_xy,
            exist_keys=exist_keys,
        )

        if has_solution:
            color = color_palette[sol_cnt % len(color_palette)]
            offset = offset_palette[sol_cnt % len(offset_palette)]

            # Update solution properties
            indices = indices.at[sol_cnt].set(i)
            colors = colors.at[sol_cnt].set(color)
            paths = paths.at[sol_cnt].set(path)
            offsets = offsets.at[sol_cnt].set(offset)
            layovers = layovers.at[sol_cnt].set(key)
            enemy_encounters = enemy_encounters.at[sol_cnt].set(enemy_encounter)

            sol_cnt += 1

        # Break if max solutions are reached
        if sol_cnt >= max_solutions:
            break

    # Trim the arrays to the actual number of solutions
    solutions = Solutions(
        n=sol_cnt,
        index=indices,
        color=colors,
        path=paths,
        offset=offsets,
        layover=layovers,
        enemy_encounter=enemy_encounters,
    )

    return solutions

@jax.jit
def get_solution_jit(env_map) -> Solutions:
    """
    Generate solutions for a scenario in a JAX-compatible way.
    """

    p_xy = jnp.argwhere(env_map == Dungeon3Tiles.PLAYER, size=1, fill_value=-1)[0]
    d_xy = jnp.argwhere(env_map == Dungeon3Tiles.DOOR, size=1, fill_value=-1)[0]

    passable_tiles = Dungeon3Problem.passable_tiles

    color_palette = generate_color_palette(30)
    offset_palette = generate_offset_palette(9)

    exist_keys = jnp.argwhere(env_map == Dungeon3Tiles.KEY, size=30, fill_value=-1)

    max_solutions = 30
    max_path_len = get_max_path_length_static(map_shape=env_map.shape)

    # Initialize arrays for solutions
    indices = jnp.full((max_solutions,), -1, dtype=jnp.int32)
    colors = jnp.zeros((max_solutions, 4), dtype=jnp.uint8)  # Assuming RGBA colors
    paths = jnp.full((max_solutions, max_path_len * 2, 2), -1, dtype=jnp.int32)
    offsets = jnp.zeros((max_solutions, 2), dtype=jnp.int32)
    layovers = jnp.full((max_solutions, 2), -1, dtype=jnp.int32)
    enemy_encounters = jnp.zeros((max_solutions, 3), dtype=jnp.int32)

    def process_key(i, carry):
        sol_cnt, indices, colors, paths, offsets, layovers, enemy_encounters = carry
        key = exist_keys[i]

        # Skip invalid keys
        def skip_invalid(_):
            return sol_cnt, indices, colors, paths, offsets, layovers, enemy_encounters

        # Process valid keys
        def process_valid(_):
            has_solution, enemy_encounter, path = check_event_jit(
                env_map=env_map,
                passable_tiles=passable_tiles,
                src=p_xy,
                key=key,
                trg=d_xy,
                exist_keys=exist_keys,
            )

            def add_solution(_):
                color = color_palette[jnp.mod(sol_cnt, color_palette.shape[0])]
                offset = offset_palette[jnp.mod(sol_cnt, offset_palette.shape[0])]

                indices_updated = indices.at[sol_cnt].set(i)
                colors_updated = colors.at[sol_cnt].set(color)
                paths_updated = paths.at[sol_cnt].set(path)
                offsets_updated = offsets.at[sol_cnt].set(offset)
                layovers_updated = layovers.at[sol_cnt].set(key)
                enemy_encounters_updated = enemy_encounters.at[sol_cnt].set(enemy_encounter)

                return (
                    sol_cnt + 1,
                    indices_updated,
                    colors_updated,
                    paths_updated,
                    offsets_updated,
                    layovers_updated,
                    enemy_encounters_updated,
                )

            return jax.lax.cond(
                has_solution,
                add_solution,
                lambda _: (sol_cnt, indices, colors, paths, offsets, layovers, enemy_encounters),
                operand=None,
            )

        return jax.lax.cond(
            jnp.all(key == jnp.array([-1, -1])),
            skip_invalid,
            process_valid,
            operand=None,
        )

    # Initial state for the loop
    initial_state = (0, indices, colors, paths, offsets, layovers, enemy_encounters)

    # Iterate over exist_keys using JAX
    sol_cnt, indices, colors, paths, offsets, layovers, enemy_encounters = jax.lax.fori_loop(
        0,
        exist_keys.shape[0],
        process_key,
        initial_state,
    )

    # Trim the arrays to the actual number of solutions
    solutions = Solutions(
        n=sol_cnt,
        index=indices,
        color=colors,
        path=paths,
        offset=offsets,
        layover=layovers,
        enemy_encounter=enemy_encounters,
    )

    return solutions


@jax.jit
def get_min_distance_jit(env_map, source_tile, target_tile, passable_tiles, max_targets: int = 5) -> int:
    """
    Compute the minimum distance from a source_tile to any of the target_tile positions.

    Args:
        env_map: 2D jnp.array, the environment map
        source_tile: int, tile ID for the source (e.g., PLAYER)
        target_tile: int, tile ID for the target (e.g., DOOR, TREASURE)
        passable_tiles: 1D jnp.array, list of passable tile IDs
        max_targets: int, maximum number of targets to consider

    Returns:
        min_distance: int
            shortest distance from source_tile to a target_tile
            -1 if no target is reachable
    """
    # Source 위치 (하나만 있다고 가정, 없으면 [-1, -1])
    src_xy = jnp.argwhere(env_map == source_tile, size=1, fill_value=-1)[0]

    # 최대 max_targets 개의 target_tile 좌표
    targets = jnp.argwhere(env_map == target_tile, size=max_targets, fill_value=-1)


    def process_target(i, carry):
        min_dist = carry
        trg = targets[i]

        def skip_invalid(_):
            return min_dist

        def process_valid(_):
            dist, _, _ = calc_path_from_a_to_b(env_map, passable_tiles, src_xy, trg)
            dist = dist.astype(jnp.int32)

            return jax.lax.cond(
                dist >= 0,
                lambda _: jnp.minimum(min_dist, dist).astype(jnp.int32),
                lambda _: min_dist.astype(jnp.int32),
                operand=None,
            )

        return jax.lax.cond(
            jnp.all(trg == jnp.array([-1, -1])),  # invalid target
            skip_invalid,
            process_valid,
            operand=None,
        )

    # 초기값: 큰 값
    init_min = int(16 ** 3)  # 4096, 맵 크기보다 충분히 큰 값

    min_distance = jax.lax.fori_loop(0, targets.shape[0], process_target, init_min)

    # reachable target이 없으면 -1 반환
    min_distance = jax.lax.cond(
        min_distance == init_min,
        lambda _: -1,
        lambda _: min_distance,
        operand=None,
    )

    return min_distance

if __name__ == '__main__':
    from debug.scenario_levels import AllLevels

    level = AllLevels[3]

    # solutions = get_solution(level)


    print(get_solution_jit(level).n)
