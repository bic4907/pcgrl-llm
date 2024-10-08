import logging


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def print_log(logger: logging.Logger, message: str, level: int = logging.DEBUG):
    # divide to multiple lines
    for line in message.split('\n'):
        logger.log(level, line)