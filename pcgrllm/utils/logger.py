import logging
from os import path
import numpy as np
import wandb
from glob import glob


from conf.config import Config
from pcgrllm.evaluation.base import EvaluationResult


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

def text_to_html(text):
    # 줄바꿈을 <br> 태그로 변환
    html_text = text.replace('\n', '<br>')

    # 탭을 4개의 공백 (&nbsp;)으로 변환
    html_text = html_text.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')

    return html_text

def log_reward_generation_data(logger, target_path: str, iteration: int):
    if wandb.run is None: return None

    # get the image files
    json_files = glob(path.join(target_path, '*.json'))
    python_files = glob(path.join(target_path, '*.py'))

    for idx, items in enumerate(zip(json_files, python_files)):
        # log the json file


        wandb.log({f'Iteration_{iteration}/reward_generation/json': wandb.Html(open(items[0], 'r').read())})
        wandb.log({f'Iteration_{iteration}/reward_generation/code': wandb.Html(text_to_html(open(items[1], 'r').read()))})


    # Log the count of json files using logger

def log_rollout_data(logger, target_path: str, iteration: int):
    if wandb.run is None: return None

    # get the images and numpy dir
    image_dir = path.join(target_path, 'images')
    numpy_dir = path.join(target_path, 'numpy')

    # get the image files
    image_files = glob(path.join(image_dir, '*.png'))
    numpy_files = glob(path.join(numpy_dir, '*.npy'))

    # log the images
    for idx, image_file in enumerate(image_files):
        # turn off wandb error

        wandb.log({f'Iteration_{iteration}/rollout/images': wandb.Image(image_file)})

    # log the numpy files, load as uint16 and log as string
    for idx, numpy_file in enumerate(numpy_files):
        numpy_data = np.load(numpy_file).astype(np.uint16)

        # make numpy string
        numpy_data = np.array2string(numpy_data, separator=',', max_line_width=10000)
        wandb.log({f'Iteration_{iteration}/rollout/numpy': wandb.Html(numpy_data)})

    # Log the count of images and numpy files using logger
    logger.info(f"Logged {len(image_files)} image files and {len(numpy_files)} numpy files to wandb for rollout.")


def log_feedback_data(logger, target_path: str, iteration: int):
    if wandb.run is None: return None

    # read json file
    json_files = glob(path.join(target_path, '*.json'))

    if len(json_files) > 0:
        json_file = json_files[0]
        if path.basename(json_file).startswith('feedback_log'):
            wandb.log({f'Iteration_{iteration}/feedback/context': wandb.Html(open(json_file, 'r').read())})

    text_files = glob(path.join(target_path, '*.txt'))

    if len(text_files) > 0:
        text_file = text_files[0]
        if path.basename(text_file) == 'feedback.txt':
            wandb.log({f'Iteration_{iteration}/feedback/response': wandb.Html(open(text_file, 'r').read())})

    # Log the count of json and text files using logger
    logger.info(f"Logged {len(json_files)} json files and {len(text_files)} text files to wandb for feedback.")

def log_evaluation_result(logger, result: EvaluationResult, iteration: int, evaluator_type: str):
    if wandb.run is None: return None

    result_dict = result.to_dict()

    for key, value in result_dict.items():
        if evaluator_type:
            log_key = f'Evaluation/{evaluator_type}/{key}'
        else:
            log_key = f'Evaluation/{key}'


        wandb.log({
            f'{log_key}': value,
            f'Evaluation/llm_iteration': iteration
        })

    # Log the evaluation result using logger
    logger.info(f"Logged evaluation result to wandb for iteration {iteration}.")
