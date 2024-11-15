import jax.numpy as jnp

from envs.pathfinding import FloodPath, calc_path_from_a_to_b, calc_path_length

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

# Initialize the FloodPath model and find the path
flood_path_net = FloodPath()
flood_path_net.init_params(env_map.shape)
# path_length, flood_count, path_coords = calc_path_from_a_to_b(
#     flood_path_net, env_map, passable_tiles, src, trg
# )

path_length, flood_count, path_coords = calc_path_from_a_to_b(
    flood_path_net, env_map, passable_tiles, src, trg
)
