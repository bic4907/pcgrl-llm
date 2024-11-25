import pygame
import hydra
import jax
from jax import numpy as jnp
import numpy as np
import sys
from conf.config import EnjoyConfig
from eval import init_config_for_eval
from utils import gymnax_pcgrl_make, init_config


@hydra.main(version_base=None, config_path='../conf', config_name='enjoy_pcgrl')
def main_enjoy(enjoy_config: EnjoyConfig):
    enjoy_config = init_config(enjoy_config)
    enjoy_config.problem = 'dungeon3'

    enjoy_config = init_config_for_eval(enjoy_config)
    env, env_params = gymnax_pcgrl_make(enjoy_config.env_name, config=enjoy_config)
    env.prob.init_graphics()

    rng = jax.random.PRNGKey(enjoy_config.eval_seed)
    obs, env_state = env.reset(rng)
    frame = env.render(env_state)  # Assuming env.render returns an RGB array
    frame_height, frame_width = frame.shape[:2]
    screen_height, screen_width = frame_height * 2, frame_width * 2

    pygame.init()

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("PCGRL Interactive Environment")
    clock = pygame.time.Clock()

    for _ in range(100):
        obs, env_state = env.reset(rng)

        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # random sample action
            rng, subkey = jax.random.split(rng)
            action = env.sample_action(subkey)

            # If an action is selected, step the environment
            if action is not None:
                action = jnp.array(action).reshape(1, 1, 1)

                obs, env_state, reward, done, info = env.step(rng, env_state, action, env_params)
                print(f"Action: {action}, Reward: {reward}, Done: {done}")

            # Render the environment to a frame
            frame = env.render(env_state)  # Assuming env.render returns an RGB array
            frame = jax.device_get(frame)  # Converts JAX Array to NumPy array
            frame = frame[..., :3]

            # Convert to uint8 if necessary
            frame = np.clip(frame, 0, 255).astype(np.uint8)

            pygame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(pygame.transform.scale(pygame_surface, (screen_width, screen_height)), (0, 0))
            pygame.display.flip()

            clock.tick(30)  # Limit FPS to 30

    pygame.quit()



if __name__ == '__main__':
    main_enjoy()
