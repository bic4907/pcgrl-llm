from copy import deepcopy

import wandb
from os import path
from conf.config import Config
from pcgrllm.utils.logger import get_wandb_name


def start_wandb(config: Config, iteration: int):
    config = deepcopy(config)

    config.exp_dir = path.join(config.exp_dir, f'iteration_{iteration}')
    config.current_iteration = iteration

    wandb.init(
        project=config.wandb_project,
        name=get_wandb_name(config),
        save_code=True)
    wandb.config.update(dict(config), allow_val_change=True)


def finish_wandb():
    if wandb.run is not None:
        wandb.finish()