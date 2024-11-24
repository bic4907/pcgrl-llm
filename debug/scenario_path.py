import jax.numpy as jnp

from envs.pathfinding import FloodPath, calc_path_from_a_to_b, calc_path_length, get_path_coords

# Example map (16x16)
env_map = jnp.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2],
    [2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2],
    [2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
])

# Define starting point (A) and target point (B)
src = jnp.array((1, 1))
trg = jnp.array((5, 5))

passable_tiles = jnp.array([1], dtype=jnp.int32)


path_length, flood_state, flood_path_net = calc_path_from_a_to_b(
    env_map, passable_tiles, src, trg
)


print('path_length', path_length)
print('flood_state', flood_state)
print('flood_path_net', flood_path_net)

path_coords = get_path_coords(flood_count=flood_state.flood_count, max_path_len=int(path_length), coord1=trg)
print('path_coords', path_coords)