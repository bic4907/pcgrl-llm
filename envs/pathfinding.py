from dataclasses import field
import math
from typing import Optional, Tuple
import numpy as np
from flax import struct
from flax.core.frozen_dict import unfreeze
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
import chex
from jax import jit, lax

from envs.utils import Tiles
from pcgrllm.utils.cuda import get_cuda_version

@struct.dataclass
class FloodPathState:
    flood_input: chex.Array
    flood_count: chex.Array
    env_map: Optional[chex.Array] = None
    trg: Optional[chex.Array] = None
    # FIXME: For some reason, we need to do this for the dungeon environment (why not maze?). Think this might be 
    #   causing a phantom path tile to render in upper-left corner of map when rendering.
    # nearest_trg_xy: Optional[chex.Array] = None #  = jnp.zeros(2, dtype=jnp.int32)
    nearest_trg_xy: Optional[chex.Array] = field(default_factory=lambda: 
                                                 (jnp.zeros(2, dtype=jnp.int32) - 1))
    done: bool = False
    has_reached_trg: bool = False


@struct.dataclass
class FloodRegionsState:
    occupied_map: chex.Array
    flood_count: chex.Array
    done: bool = False


# FIXME: It's probably definitely (?) inefficient to use NNs here. We should use `jax.lax.convolve` directly.
#   (Also would allow us to use ints instead of floats?)
class FloodPath(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(1, kernel_size=(3, 3), padding='SAME', kernel_init=constant(0.0), bias_init=constant(0.0))(x)
        return x

    def init_params(self, map_shape):
        rng = jax.random.PRNGKey(0) # This key doesn't matter since we'll reset before playing anyway(?)
        init_x = jnp.zeros(map_shape + (2,), dtype=jnp.float32)
        self.flood_params = unfreeze(self.init(rng, init_x))
        flood_kernel = self.flood_params['params']['Conv_0']['kernel']
        # Walls on center tile prevent it from being flooded
        flood_kernel = flood_kernel.at[1, 1, 0].set(-5)
        # Flood at adjacent tile produces flood toward center tile
        flood_kernel = flood_kernel.at[1, 2, 1].set(1)
        flood_kernel = flood_kernel.at[2, 1, 1].set(1)
        flood_kernel = flood_kernel.at[1, 0, 1].set(1) 
        flood_kernel = flood_kernel.at[0, 1, 1].set(1)
        flood_kernel = flood_kernel.at[1, 1, 1].set(1)
        self.flood_params['params']['Conv_0']['kernel'] = flood_kernel

    def flood_step(self, flood_state: FloodPathState):
        """Flood until no more tiles can be flooded."""
        flood_input, flood_count = flood_state.flood_input, flood_state.flood_count
        flood_params = self.flood_params
        occupied_map = flood_input[..., 0]
        flood_out = self.apply(flood_params, flood_input)
        flood_out = jnp.clip(flood_out, a_min=0, a_max=1)
        flood_out = jnp.stack([occupied_map, flood_out[..., -1]], axis=-1)
        flood_count = flood_out[..., -1] + flood_count
        done = jnp.all(flood_input == flood_out)
        flood_state = FloodPathState(flood_input=flood_out, flood_count=flood_count, done=done)
        return flood_state

    def flood_step_trg(self, flood_state: FloodPathState):
        """Flood until a target tile type is reached."""
        flood_input, flood_count = flood_state.flood_input, flood_state.flood_count
        trg = flood_state.trg
        flood_params = self.flood_params
        occupied_map = flood_input[..., 0]
        flood_out = self.apply(flood_params, flood_input)
        flood_out = jnp.clip(flood_out, a_min=0, a_max=1)
        flood_out = jnp.stack([occupied_map, flood_out[..., -1]], axis=-1)
        flood_count = flood_out[..., -1] + flood_count
        nearest_trg_xy = jnp.argwhere(
                jnp.where(flood_state.env_map == trg, flood_count, 0) > 0,
            size=1, fill_value=-1)[0]
        has_reached_trg = jnp.logical_not(jnp.all(nearest_trg_xy == -1))
        no_trg = jnp.all(flood_state.env_map != trg)
        no_change = jnp.all(flood_input == flood_out)
        done = has_reached_trg | no_trg | no_change
        flood_state = FloodPathState(flood_input=flood_out, flood_count=flood_count, done=done,
                                     env_map=flood_state.env_map, trg=trg,
                                     nearest_trg_xy=nearest_trg_xy)

        return flood_state

    def flood_step_trg_cell(self, flood_state: FloodPathState):
        """Flood until a target tile type is reached."""
        flood_input, flood_count = flood_state.flood_input, flood_state.flood_count

        trg_x, trg_y = flood_state.trg[1], flood_state.trg[0] # check x, y

        flood_params = self.flood_params
        occupied_map = flood_input[..., 0]
        flood_out = self.apply(flood_params, flood_input)
        flood_out = jnp.clip(flood_out, a_min=0, a_max=1)
        flood_out = jnp.stack([occupied_map, flood_out[..., -1]], axis=-1)

        flood_count = flood_out[..., -1] + flood_count

        has_reached_trg = jax.numpy.where(flood_count[trg_y, trg_x] > 0, True, False)
        no_change = jnp.all(flood_input == flood_out)
        done = has_reached_trg | no_change

        # make zeros env_map and mark at trg_cell
        # nearest_trg_xy = flood_state.trg
        trg_arr = jnp.zeros(flood_state.env_map.shape)
        trg_arr = trg_arr.at[trg_y, trg_x].set(1)

        nearest_trg_xy = jnp.argwhere(
            jnp.where(trg_arr, flood_count, 0) > 0,
            size=1, fill_value=-1)[0]

        flood_state = FloodPathState(flood_input=flood_out, flood_count=flood_count, done=done,
                                     env_map=flood_state.env_map, trg=flood_state.trg,
                                     has_reached_trg=has_reached_trg,
                                     nearest_trg_xy=nearest_trg_xy)

        return flood_state


class FloodRegions(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(5, kernel_size=(3, 3), padding='SAME', kernel_init=constant(0.0), bias_init=constant(0.0))(x)
        return x

    def init_params(self, map_shape):
        rng = jax.random.PRNGKey(0)  # This key doesn't matter since we'll reset before playing anyway(?)
        init_x = jnp.zeros(map_shape + (1,), dtype=jnp.float32)
        self.flood_params = unfreeze(self.init(rng, init_x))
        flood_kernel = self.flood_params['params']['Conv_0']['kernel']
        flood_kernel = flood_kernel.at[1, 1, 0, 0].set(1)
        flood_kernel = flood_kernel.at[1, 2, 0, 1].set(1)
        flood_kernel = flood_kernel.at[2, 1, 0, 2].set(1)
        flood_kernel = flood_kernel.at[1, 0, 0, 3].set(1)
        flood_kernel = flood_kernel.at[0, 1, 0, 4].set(1)
        self.flood_params['params']['Conv_0']['kernel'] = flood_kernel

    def flood_step(self, flood_regions_state: FloodRegionsState):
        """Flood until no more tiles can be flooded."""
        occupied_map, flood_count = flood_regions_state.occupied_map, flood_regions_state.flood_count
        flood_params = self.flood_params
        flood_out = self.apply(flood_params, flood_count)
        flood_count = jnp.max(flood_out, -1, keepdims=True)
        flood_count = flood_count * (1 - occupied_map[..., None])
        done = jnp.all(flood_count == flood_regions_state.flood_count)
        flood_regions_state = FloodRegionsState(flood_count=flood_count,
                                                occupied_map=occupied_map, done=done)
        return flood_regions_state

    # def flood_step(self, flood_state: FloodRegionsState, unused):
    #     done = flood_state.done
    #     # flood_state = jax.lax.cond(done, lambda _: flood_state, self._flood_step, flood_state)
    #     flood_state = self._flood_step(flood_state)
    #     return flood_state, None

    def flood_step_while(self, flood_state: FloodRegionsState):
        flood_state, _ = self.flood_step(flood_state=flood_state, unused=None)
        return flood_state

        
def get_path_coords(flood_count: chex.Array, max_path_len, coord1):
    y, x = coord1
    path_coords = jnp.full((max_path_len, 2), fill_value=-1, dtype=jnp.int32)
    path_coords = path_coords.at[0].set((y, x))
    curr_val = flood_count[y, x]
    max_val = jnp.max(flood_count)
    flood_count = jnp.where(flood_count == 0, jnp.inf, flood_count)
    padded_flood_count = jnp.pad(flood_count, 1, mode='constant', constant_values=jnp.inf)

    # while curr_val < max_val:
    def get_next_coord(carry):
        # Get the coordinates of a neighbor tile where the count is curr_val + 1
        curr_val, path_coords, padded_flood_count, i = carry
        last_yx = path_coords[i]
        padded_flood_count = padded_flood_count.at[last_yx[0] + 1, last_yx[1] + 1].set(jnp.inf)
        # nb_slice = padded_flood_count.at[last_xy[0]-1:last_xy[0] + 2, last_xy[1]-1:last_xy[1] + 2]
        nb_slice = jax.lax.dynamic_slice(padded_flood_count, (last_yx[0], last_yx[1]), (3, 3))
        # Mask out the corners to prevent diagonal movement
        nb_slice = nb_slice.at[[0, 0, 2, 2], [0, 2, 0, 2]].set(jnp.inf)
        # Get the coordinates of the minimum value
        y, x = jnp.argwhere(nb_slice == curr_val + 1, size=1)[0]
        y, x = y - 1, x - 1
        yx = last_yx + jnp.array([y, x])
        path_coords = path_coords.at[i + 1].set(yx)
        return curr_val + 1, path_coords, padded_flood_count, i + 1

    def cond(carry):
        curr_val, _, _, _ = carry
        return curr_val < max_val

    # path_coords = jax.lax.scan(get_next_coord, None, length=int(max_val - curr_val))
    _, path_coords, _, _ = jax.lax.while_loop(cond, get_next_coord, (curr_val, path_coords, padded_flood_count, 0))

    # while curr_val < max_val:
    #     get_next_coord()

    return path_coords


def get_path_coords_diam(flood_count: chex.Array, max_path_len):
    """Get the path coordinates from the flood count."""
    # Get the coordinates of a tile where the count is 1
    yx = jnp.unravel_index(jnp.argmin(jnp.where(flood_count == 0, jnp.inf, flood_count)), flood_count.shape)
    return get_path_coords(flood_count, max_path_len, yx)


def get_max_path_length(map_shape: Tuple[int]):
    map_shape = jnp.array(map_shape)
    return (jnp.ceil(jnp.prod(map_shape) / 2) + jnp.max(map_shape)).astype(int)


def get_max_path_length_static(map_shape: Tuple[int]):
    return int(math.ceil(math.prod(map_shape) / 2) + max(map_shape))


def get_max_n_regions(map_shape: Tuple[int]):
    map_shape = jnp.array(map_shape)
    return jnp.ceil(jnp.prod(map_shape) / 2).astype(int)


def get_max_n_regions_static(map_shape: Tuple[int]):
    return int(math.ceil(math.prod(map_shape) / 2))


def calc_n_regions(flood_regions_net: FloodRegions, env_map: chex.Array, passable_tiles: chex.Array):
    """Approximate the diameter of a maze-like tile map."""
    max_path_length = get_max_path_length_static(env_map.shape)
    max_n_regions = get_max_n_regions_static(env_map.shape)

    # Get array of flattened indices of all tiles in env_map
    idxs = jnp.arange(math.prod(env_map.shape), dtype=jnp.float32) + 1
    regions_flood_count = idxs.reshape(env_map.shape)

    # Mask out flood_count where env_map is not empty
    occupied_map = (env_map[..., None] != passable_tiles).all(-1)
    # occupied_map = env_map != Tiles.EMPTY
    init_flood_count = regions_flood_count * (1 - occupied_map)

    flood_regions_state = FloodRegionsState(
        flood_count=init_flood_count[..., None],
        occupied_map=occupied_map, done=False)
    # flood_regions_state, _ = jax.lax.scan(flood_regions_net.flood_step, flood_regions_state, jnp.arange(max_path_length))

    # FIXME: Check if the while_loop doesn't work in CUDA11
    flood_regions_state = jax.lax.while_loop(
        lambda frs: jnp.logical_not(frs.done),
        flood_regions_net.flood_step,
        flood_regions_state)
    regions_flood_count = flood_regions_state.flood_count.astype(jnp.int32)

    # FIXME: Sketchily upper-bounding number of regions here since we need a concrete value
    n_regions = jnp.clip(
        jnp.unique(regions_flood_count, size=max_n_regions, fill_value=0),
        0, 1).sum()

    return n_regions, regions_flood_count[..., 0]


def calc_path_length(flood_path_net, env_map: jnp.ndarray, passable_tiles: jnp.ndarray, src: int, trg: chex.Array):
    occupied_map = (env_map[..., None] != passable_tiles).all(-1).astype(jnp.float32)
    init_flood = (env_map == src).astype(jnp.float32)
    init_flood_count = init_flood.copy()
    # Concatenate init_flood with new_occ_map
    flood_input = jnp.stack([occupied_map, init_flood], axis=-1)
    flood_state = FloodPathState(flood_input=flood_input, flood_count=init_flood_count, env_map=env_map, trg=trg,
                                 done=False)
    flood_state = jax.lax.while_loop(
        lambda fps: jnp.logical_not(fps.done),
        flood_path_net.flood_step_trg,
        flood_state)
    path_length = jnp.clip(
        flood_state.flood_count.max() - jnp.where(
            (flood_state.flood_count == 0) | (env_map != trg), jnp.inf, flood_state.flood_count).min(),
        0)
    return path_length, flood_state.flood_count, flood_state.nearest_trg_xy


def calc_diameter(flood_regions_net: FloodRegions, flood_path_net: FloodPath, env_map: chex.Array,
                  passable_tiles: chex.Array):
    """Approximate the diameter of a maze-like tile map. Simultaneously compute
    the number of regions (connected traversible components) in the map."""
    max_path_length = get_max_path_length_static(env_map.shape)
    max_n_regions = get_max_n_regions_static(env_map.shape)

    # Get array of flattened indices of all tiles in env_map
    idxs = jnp.arange(math.prod(env_map.shape), dtype=jnp.float32) + 1
    regions_flood_count = idxs.reshape(env_map.shape)

    # Mask out flood_count where env_map is not empty
    occupied_map = (env_map[..., None] != passable_tiles).all(-1)
    init_flood_count = regions_flood_count * (1 - occupied_map)

    # We'll use this for determining region anchors later
    pre_flood_count = (regions_flood_count + 1) * (1 - occupied_map)

    flood_regions_state = FloodRegionsState(flood_count=init_flood_count[..., None], occupied_map=occupied_map)
    # flood_regions_state, _ = jax.lax.scan(flood_regions_net.flood_step, flood_regions_state, jnp.arange(max_path_length))
    flood_regions_state = jax.lax.while_loop(
        lambda frs: jnp.logical_not(frs.done),
        flood_regions_net.flood_step,
        flood_regions_state)
    regions_flood_count = flood_regions_state.flood_count.astype(jnp.int32)

    # FIXME: Sketchily upper-bounding number of regions here since we need a concrete value
    n_regions = jnp.clip(
        jnp.unique(regions_flood_count, size=max_n_regions, fill_value=0),
        0, 1).sum()
    idxs = jnp.arange(math.prod(env_map.shape), dtype=jnp.float32)
    # path_length, flood_path_state = self.calc_path(env_map)
    regions_flood_count = flood_regions_state.flood_count[..., 0]

    # Because the maximum index value of each region has flooded throughout, we can select the tile at which this
    # maximum index value originally appeared as the "anchor" of this region.
    region_anchors = jnp.clip(pre_flood_count - regions_flood_count, 0, 1)

    # Now we flood out from all of these anchor points and find the furthest point from each.
    init_flood = region_anchors
    init_flood_count = init_flood
    flood_input = jnp.stack([occupied_map, init_flood], axis=-1)
    flood_path_state = FloodPathState(flood_input=flood_input, flood_count=init_flood_count)
    # flood_path_state, _ = jax.lax.scan(flood_path_net.flood_step, flood_path_state, None, max_path_length)
    flood_path_state = jax.lax.while_loop(
        lambda state: jnp.logical_not(state.done),
        flood_path_net.flood_step,
        flood_path_state)

    # We need to find the max path length in *each region*. So we'll mask out the path lengths of all other regions.
    # Unique (max) region indices
    region_idxs = jnp.unique(regions_flood_count, size=max_n_regions + 1, fill_value=0)[
                  1:]  # exclude the `0` non-region
    region_masks = jnp.where(regions_flood_count[..., None] == region_idxs, 1, 0)
    path_flood_count = flood_path_state.flood_count
    region_path_floods = path_flood_count[..., None] * region_masks
    region_path_floods = region_path_floods.reshape((-1, region_path_floods.shape[-1]))
    # Now we identify the furthest point from the anchor in each region.
    region_endpoint_idxs = jnp.argmin(jnp.where(region_path_floods == 0, max_path_length * 2, region_path_floods),
                                      axis=0)
    region_endpoint_idxs = jnp.unravel_index(region_endpoint_idxs, env_map.shape)

    # Because we likely have more region endpoint indices than actual regions,
    # we have many empty region floods, and so many false (0, 0) region
    # endpoints. We'll mask these out by converting them to (-1, -1) then
    # padding (then cropping) the bottom/right of the init flood.
    valid_regions = region_path_floods.sum(0) > 0
    region_endpoint_idxs = (
        jnp.where(valid_regions, region_endpoint_idxs[0], -1),
        jnp.where(valid_regions, region_endpoint_idxs[1], -1),
    )
    init_flood = jnp.zeros(np.array(env_map.shape) + 1)
    init_flood = init_flood.at[region_endpoint_idxs].set(1)
    init_flood = init_flood[:-1, :-1]

    # We can now flood out from these endpoints.
    flood_input = jnp.stack([occupied_map, init_flood], axis=-1)
    flood_path_state = FloodPathState(flood_input=flood_input, flood_count=init_flood)
    # flood_path_state, _ = jax.lax.scan(flood_path_net.flood_step, flood_path_state, None, max_path_length)
    flood_path_state = jax.lax.while_loop(
            lambda state: jnp.logical_not(state.done),
            flood_path_net.flood_step,
            flood_path_state)
    path_length = jnp.clip(flood_path_state.flood_count.max() - jnp.where(flood_path_state.flood_count == 0, max_path_length, flood_path_state.flood_count).min(), 0)

    return path_length, flood_path_state, n_regions, flood_regions_state


def calc_path_from_a_to_b(env_map: chex.Array,
                          passable_tiles: chex.Array,
                          src: chex.Array, trg: chex.Array,
                          flood_path_net: FloodPath = None) -> Tuple[int, chex.Array, chex.Array]:


    if flood_path_net is None:
        flood_path_net = FloodPath()
        flood_path_net.init_params(env_map.shape)

    occupied_map = (env_map[..., None] != passable_tiles).all(-1).astype(jnp.float32)

    src_y, src_x = src

    init_flood = jnp.zeros_like(env_map, dtype=jnp.float32)
    init_flood = init_flood.at[src_y, src_x].set(1)

    init_flood_count = init_flood.copy()

    flood_input = jnp.stack([occupied_map, init_flood], axis=-1)
    flood_state = FloodPathState(flood_input=flood_input, flood_count=init_flood_count, env_map=env_map,
                                 trg=trg,
                                 done=False)


    CUDA_VERSION = get_cuda_version()
    if CUDA_VERSION is None or CUDA_VERSION >= 12:
        # Use `scan` for CUDA 12 or higher
        # Use `while_loop` for CUDA 11 or lower
        @jit
        def is_search_done(flood_state):
            return jnp.logical_not(flood_state.done)

        flood_state = lax.while_loop(
            is_search_done,
            flood_path_net.flood_step_trg_cell,
            flood_state
        )
    else:
        @jit
        def flood_body(flood_state, _):
            new_state = lax.cond(
                flood_state.done,
                lambda state: state,  # 상태 유지
                lambda state: flood_path_net.flood_step_trg_cell(state),  # 업데이트
                flood_state
            )
            return new_state, None  # scan은 반드시 (state, output) 형식을 반환

        max_steps = env_map.size
        flood_state, _ = lax.scan(flood_body, flood_state, xs=None, length=max_steps)


    path_length = jnp.clip(
        flood_state.flood_count.max() - jnp.where(
            (flood_state.flood_count == 0), jnp.inf, flood_state.flood_count).min(),
        0)

    has_reached_trg = flood_state.has_reached_trg
    path_length = jnp.where(has_reached_trg, path_length, -1.0)

    return path_length, flood_state, flood_path_net


def check_event(env_map: chex.Array,
                passable_tiles: chex.Array,
                src: chex.Array, key: chex.Array, trg: chex.Array,
                flood_path_net: FloodPath = None) -> Tuple[int, chex.Array, chex.Array]:
    max_path_len = get_max_path_length_static(map_shape=env_map.shape)

    solutions = []
    pk_path_length, pk_flood_state, _ = calc_path_from_a_to_b(env_map, passable_tiles, src, key)
    kd_path_length, kd_flood_state, _ = calc_path_from_a_to_b(env_map, passable_tiles, key, trg)

    if pk_path_length > 0 and kd_path_length > 0:
        whole_path = get_whole_path(pk_flood_state, kd_flood_state, max_path_len)
        solutions.append(whole_path)

    copy_env_map = env_map.copy()

    # erase Monster
    for monster_label in [4, 5, 6]:
        copy_env_map, solution = get_len_solutions(copy_env_map, passable_tiles, src, key, trg, monster_label)
        if solution in solutions:
            continue
        solutions += solution
    if len(solutions) != 0:
        solutions = remove_duplicates(solutions)
        return len(solutions), solutions
    else:
        return 0, None

def remove_duplicates(data):
    unique_data = []
    seen = []

    for item in data:
        key1 = item[0].flatten()
        key2 = item[1].flatten()

        # check duplicate
        if not any(jnp.array_equal(key1, s[0]) and jnp.array_equal(key2, s[1]) for s in seen):
            seen.append((key1, key2))
            unique_data.append(item)

    return unique_data

def get_len_solutions(env_map: chex.Array,
                      passable_tiles: chex.Array,
                      src: chex.Array, key: chex.Array, trg: chex.Array,
                      monster_label: chex.Array) -> Tuple[int, chex.Array]:
    solutions = []
    max_path_len = get_max_path_length_static(map_shape=env_map.shape)

    env_map = jnp.array(env_map)
    for _xy in jnp.argwhere(env_map == monster_label):
        y, x = _xy

        env_map.at[y, x].set(1)
        pk_path_length, pk_flood_state, _ = calc_path_from_a_to_b(env_map, passable_tiles, src, key)
        kd_path_length, kd_flood_state, _ = calc_path_from_a_to_b(env_map, passable_tiles, key, trg)
        if pk_path_length > 0 and kd_path_length > 0:
            whole_path = get_whole_path(pk_flood_state, kd_flood_state, max_path_len)
            solutions.append(whole_path)

    return env_map, solutions

def get_whole_path(pk_flood_state: chex.Array, kd_flood_state: chex.Array, max_path_len: chex.Array) -> Tuple[int, chex.Array]:
    pk_flood_count = pk_flood_state.flood_count
    kd_flood_count = kd_flood_state.flood_count
    pk_coords = get_path_coords(pk_flood_count, max_path_len=max_path_len, coord1=pk_flood_state.trg)
    kd_coords = get_path_coords(kd_flood_count, max_path_len=max_path_len, coord1=kd_flood_state.trg)

    return (pk_coords, kd_coords)
