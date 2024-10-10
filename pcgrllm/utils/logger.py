import logging

from conf.config import Config


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def print_log(logger: logging.Logger, message: str, level: int = logging.DEBUG):
    # divide to multiple lines
    for line in message.split('\n'):
        logger.log(level, line)

def get_wandb_name(config: Config):
    exp_dir_path = config.exp_dir
    # split by directory
    exp_dirs = exp_dir_path.split('/')

    # if the last directory includes "iteration" concat the last two directories
    if "iteration" in exp_dirs[-1]:
        return ":".join(exp_dirs[-2:])
    else:
        return exp_dirs[-1]

