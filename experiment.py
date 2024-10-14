from distutils.command.config import config
from os.path import basename, dirname

import hydra
import json
import os
import shutil
import copy
from datetime import datetime
from os import path
import subprocess
import argparse

import requests
import yaml
import tempfile
import threading
import platform
import pandas as pd
import logging
import os
import site
import platform
import pprint

from scipy.stats import dweibull

from conf.config import TrainConfig
from eval import main_eval

from pcgrllm.utils.logger import print_log
from pcgrllm.utils.path_utils import init_config

from pcgrllm.validate_reward import run_validate
from pcgrllm.stage import Stage


import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="orbax")
warnings.filterwarnings("ignore", category=FutureWarning, module="jax")

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))


logging.getLogger('hydra').setLevel(logging.INFO)
logging.getLogger('absl').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)


class Experiment:

    def __init__(self, config: TrainConfig):
        self.config = config

        if config.overwrite and os.path.exists(self.config.exp_dir):
            shutil.rmtree(self.config.exp_dir)

        os.makedirs(self.config.exp_dir, exist_ok=True)

        self._setup_logger()
        self.initialize()


        self.logging(pprint.pformat(self.config, indent=4), level=logging.INFO)

    def initialize(self):
        self._iteration = 1
        self._stage = Stage.Analysis
        self._current_reward_function_filename = None

        self._copy_prompt()

    @property
    def _experiment_path(self):
        return self.config.exp_dir

    def log_with_prefix(self, message, level=logging.DEBUG):
        """Logs a message with a formatted prefix."""
        info_dict = {
            'outer_loop': getattr(self, '_iteration', -1),
        }

        # Define the prefix format
        prefix = '[#iter: {outer_loop}]'.format(**info_dict)

        # Split the message by line breaks and log each line with the prefix
        message = str(message)
        for line in message.splitlines():
            formatted_message = f'{prefix} {line}'
            self.logger.log(level, formatted_message)


    @property
    def reward_function_log_path(self):
        return path.join(self._experiment_path, self._reward_function_log_filename)


    def _copy_prompt(self):
        """Copies the prompt log file to the experiment directory."""

        self.logging("Copying prompt directory to the experiment directory", level=logging.INFO)

        source_dir = path.join(path.dirname(__file__), 'pcgrllm', 'prompt')
        dest_dir = path.join(self._experiment_path, 'prompt')

        self.logging(f"Copying prompt directory to the experiment directory: {source_dir} -> {dest_dir}")

        try:
            shutil.copytree(source_dir, dest_dir)
        except FileExistsError:
            self.logging(f"Prompt directory already exists: {dest_dir}")
            pass

    @property
    def reward_functions_dir(self):
        return path.join(self._experiment_path, 'reward_functions')

    @property
    def feedback_dir(self):
        return path.join(self._experiment_path, 'feedbacks')


    def _create_reward_functions_dir(self):
        try:
            os.makedirs(self.reward_functions_dir, exist_ok=self._resume or self._overwrite)
        except FileExistsError:
            self.logging(f"Reward functions directory already exists: {self.reward_functions_dir}")
            raise

    def _create_feedback_dir(self):
        try:
            os.makedirs(self.feedback_dir, exist_ok=self._resume or self._overwrite)
        except FileExistsError:
            self.logging(f"Feedback directory already exists: {self.feedback_dir}")
            raise

    def _bypass_reward_function(self) -> str:
        raise NotImplementedError


    def generate_reward_function(self):
        """Generates a reward function using the reward generator script."""
        self.logging(f"Generating reward function for iteration {self._iteration}", level=logging.INFO)


        args_dict = {
            'shared_storage_path': self._experiment_path,
            'postfix': f"reward_outer_{self._iteration}",
            'reward_functions_dir': 'reward_functions',
            'gpt_model': self.config.gpt_model,
            'gpt_max_token': 4096,
            'verbose': None,
            'previous_reward_function': self._current_reward_function_filename,
            'trial_count': self.config.n_generation_trials,
            'n_outer': self._iteration,
            'n_inner': 1,
            # 'reference_csv': self._reference_csv,
            'iteration_num': self._iteration,
            'feedback_path': None
        }

        self.logging(f"Reward generation arguments:\n{pprint.pformat(args_dict, indent=4)}", level=logging.INFO)


        # Start of the 'generate_reward.py'
        from pcgrllm.generate_reward import generate_reward
        reward_name = generate_reward(config=self.config, generate_args=args_dict)

        return reward_name

    def bypass_reward_function(self):
        # copy the reward function

        os.makedirs(self.reward_functions_dir, exist_ok=True)

        reward_filename = f'{basename(self.config.bypass_reward_path)}.py'
        origin_path = path.join(dirname(__file__), 'pcgrllm', 'example', reward_filename)
        target_path = path.join(self.reward_functions_dir, reward_filename)

        self.logging(f"Copying reward function to the experiment directory: {origin_path} -> {target_path}", logging.WARNING)
        shutil.copy(origin_path, target_path)

        return target_path

    def validate_reward_function(self):
        config = copy.deepcopy(self.config)

        try:
            result = run_validate(config)
        except:
            result = False

        print_log(self.logger, f"Reward validation passed?: {result}", level=logging.INFO)

        return result

    def append_reward_generation_log(self, result: str, trial_num: int, previous_reward_function: str,
                                     current_reward_function: str) -> None:
        # append dataframe
        row = {
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            'Iteration': self._iteration,
            'InstanceUUID': platform.node(),
            'Trial': trial_num,
            'Result': result,
            'PreviousFileName': previous_reward_function,
            'CurrentFileName': current_reward_function,
            'Academy.TotalStepCount': 0,
            'Academy.EpisodeCount': 0
        }

        try:
            df = pd.read_csv(self.reward_function_log_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        except FileNotFoundError:
            df = pd.DataFrame([row])

        df.to_csv(self.reward_function_log_path, index=False)

    def train_pcgrl(self):
        """Runs the mlagents-learn command with the given parameters."""

        config = copy.deepcopy(self.config)
        config.exp_dir = path.join(config.exp_dir, f'iteration_{self._iteration}')
        config.initialize = False
        config.llm_iteration = self._iteration


        config.reward_function_path = self._current_reward_function_filename
        # copy to the exp_dir

        os.makedirs(config.exp_dir, exist_ok=True)

        self.logging(f'Copying reward function to the experiment directory: {config.reward_function_path} -> {path.join(config.exp_dir, basename(self._current_reward_function_filename))}')
        shutil.copy(config.reward_function_path, path.join(config.exp_dir, basename(self._current_reward_function_filename)))
        config.reward_function_path = path.join(config.exp_dir, basename(self._current_reward_function_filename))

        config.overwrite = False

        from train import main as train

        train(config)

        return True


    def rollout_pcgrl(self, iteration_run_id):
        from eval import main_eval as run_eval

        config = copy.deepcopy(self.config)
        config.initialize = False
        config.exp_dir = path.join(config.exp_dir, 'iteration_' + str(iteration_run_id))
        config.random_agent = False
        config.llm_iteration = self._iteration

        run_eval(config)


    # 파일 분석
    def analyze_output(self, iteration_num: int) -> None:

        from pcgrllm.generate_feedback import generate_feedback

        args_dict = {
            'exp_path': self.config.exp_dir,
            # 'exp_path': self.config.exp_dir,
            'condition_prompt': 'Make a level looks like "A"',
            'exp_path': '/Users/inchang/Desktop/pcgrl-llm/pcgrllm/example/binary_narrow-w-16_gpt_model-gpt-4o-gil_3',
            'input_type': 'array',
            'gpt_model': self.config.gpt_model,
            'reward_function': self._current_reward_function_filename,
            'iteration': self._iteration,
        }

        feedback = generate_feedback(self.config, args_dict)

        self.logging
        self.feedback = feedback

        return True

    def save_state(self):
        # target variables: iteration, current_reward_function_filename
        serialize_items = ['_stage', '_iteration', '_current_reward_function_filename']

        with open(path.join(self._experiment_path, 'state.yaml'), 'w') as file:
            yaml.dump({item: getattr(self, item) for item in serialize_items}, file)


    def load_state(self):
        self.logging("Loading state from the previous experiment.")

        try:
            with open(path.join(self._experiment_path, 'state.yaml'), 'r') as file:
                state = yaml.safe_load(file)

                for key, value in state.items():
                    setattr(self, key, value)
        except FileNotFoundError:
            self.logging("State file not found. Exiting.")

    def run(self):

        self.logging("Running experiment", level=logging.DEBUG)

        while not self._stage is Stage.Done:

            self.logging(f"Current stage: {self._stage}", level=logging.DEBUG)

            if self._stage == Stage.StartIteration:
                self._stage = Stage.RewardGeneration



            elif self._stage == Stage.RewardGeneration:

                if self.config.bypass_reward_path is not None:
                    reward_generation_fn = self.bypass_reward_function
                else:
                    reward_generation_fn = self.generate_reward_function

                self._current_reward_function_filename = reward_generation_fn()

                if self._current_reward_function_filename is False:
                    self.exit("Reward function generation failed. Exiting.")
                else:
                    self._stage = Stage.TrainPCGRL


            elif self._stage == Stage.TrainPCGRL:
                # Run ML-Agents
                self.train_pcgrl()

                self._stage = Stage.RolloutPCGRL
            elif self._stage == Stage.RolloutPCGRL:
                # Collect results
                self.rollout_pcgrl(self._iteration)

                self._stage = Stage.Analysis
            elif self._stage == Stage.Analysis:
                # Analyze results
                self.analyze_output(self._iteration) # TODO
                self._stage = Stage.FinishIteration

            elif self._stage == Stage.FinishIteration:

                if self._iteration >= self.config.total_iterations:
                    self._stage = Stage.Done
                else:
                    self._iteration += 1
                    self._stage = Stage.StartIteration

            self.save_state()

        self.logging("Experiment finished.")

    def exit(self, message: str, code: int = 1):
        self.logging(message, level=logging.ERROR)
        exit(code)

    def _setup_logger(self):
        """Sets up the logger for the experiment."""
        self.logger = logging.getLogger(basename(__file__))
        self.logger.setLevel(logging.DEBUG)


    def logging(self, message, level=logging.DEBUG):
        info_dict = {
            'outer_loop': self._iteration if hasattr(self, '_iteration') else -1,
            'total_iterations': self.config.total_iterations,
        }

        # Define the prefix format
        prefix = '[iter: {outer_loop}/{total_iterations}]'.format(**info_dict)

        # Split the message by line breaks and log each line with the prefix
        message = str(message)
        for line in message.splitlines():
            formatted_message = f'{prefix} {line}'
            self.logger.log(level, formatted_message)



@hydra.main(version_base=None, config_path='./conf', config_name='train_pcgrllm')
def main(config: TrainConfig):
    init_config(config)

    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    main()

