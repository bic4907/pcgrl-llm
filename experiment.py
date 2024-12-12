import random
from distutils.command.config import config
from glob import glob
from os.path import basename, dirname
from typing import Optional

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

from conf.config import TrainConfig
from envs.pcgrl_env import get_prob_cls, ProbEnum, get_available_tiles
from pcgrllm.evaluation.base import EvaluationResult
from pcgrllm.evaluation.heuristic import HeuristicEvaluator
from pcgrllm.evaluation.llm_evaluator import LLMEvaluator
from pcgrllm.evaluation.solution import SolutionEvaluator
from pcgrllm.evaluation.vit import ViTEvaluator
from pcgrllm.scenario_preset import Scenario, ScenarioPreset
from pcgrllm.task import TaskType
from pcgrllm.utils.graph import GraphManager, NodeInfo

from pcgrllm.utils.logger import print_log, log_rollout_data, log_feedback_data, log_reward_generation_data, \
    log_evaluation_result
from pcgrllm.utils.path_utils import init_config
from pcgrllm.utils.prompt import get_reward_score_paired_examples
from pcgrllm.utils.storage import Iteration, Storage
from pcgrllm.utils.wandb import start_wandb, finish_wandb
from pcgrllm.generate_feedback import generate_feedback

from pcgrllm.validate_reward import run_validate
from pcgrllm.stage import Stage


import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="orbax")
warnings.filterwarnings("ignore", category=FutureWarning, module="jax")

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
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

        self._stage = Stage.StartIteration
        self._current_reward_function_filename = None
        self._current_feedback_path = None
        self.previous_reward_function_path = None
        self.previous_feedback_path = None
        self.auxiliary_iter_nums = list()

        self.storage = Storage(self.config.exp_dir)
        self.graph_manager = GraphManager(storage=self.storage, max_breadth=self.config.branch_factor)

        self.load_state()

        self._copy_prompt()
        self._copy_scenario()

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

    def _copy_scenario(self):
        """Copies the scenario log file to the experiment directory."""

        self.logging("Copying scenario directory to the experiment directory", level=logging.INFO)

        source_dir = path.join(path.dirname(__file__), 'pcgrllm', 'scenario')
        dest_dir = path.join(self._experiment_path, 'scenario')

        self.logging(f"Copying scenario directory to the experiment directory: {source_dir} -> {dest_dir}")

        try:
            shutil.copytree(source_dir, dest_dir)
        except FileExistsError:
            self.logging(f"Scenario directory already exists: {dest_dir}")
            pass

    @property
    def reward_functions_dir(self):
        return path.join(self._experiment_path, 'reward_functions')

    @property
    def feedback_dir(self):
        return path.join(self._experiment_path, f'iteration_{self._iteration}', 'feedbacks')


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

    def generate_reward_function(self):
        """Generates a reward function using the reward generator script."""
        self.logging(f"Generating reward function for iteration {self._iteration}", level=logging.INFO)

        # check if there is previous iteration,
        prev_eval_result, auxiliary_prompt_path = None, None

        if self.config.pe in ['got'] and self._iteration >= 2:
            prev_eval_result = self.get_evaluation_result(self._iteration - 1).to_prompt()
            self.auxiliary_iter_nums = self.graph_manager.get_best_iteration_nums(max_n=2, excludes=[self._iteration - 1])

            auxiliary_prompt = get_reward_score_paired_examples(self.storage, self.auxiliary_iter_nums)
            auxiliary_prompt_path = self.storage.set_auxiliary_prompt(self._iteration, auxiliary_prompt)

        target_character = self.config.target_character

        if self.config.task == 'scenario':

            # if the self.config.target_character is numberic value, then use the scenario prompt

            # the target_character is basically a string, try to convert it to int and check if it is a number
            if self.config.target_character.isnumeric():
                scenario_preset = ScenarioPreset()
                scenario_preset = scenario_preset.scenarios.get(target_character, None)
                target_character = scenario_preset.prompt if scenario_preset is not None else target_character

        args_dict = {
            'shared_storage_path': self._experiment_path,
            'postfix': f"reward_outer_{self._iteration}",
            'reward_functions_dir': 'reward_functions',
            'gpt_model': self.config.gpt_model,
            'gpt_max_token': 4096,
            'verbose': None,
            'previous_reward_function': self.previous_reward_function_path,
            'trial_count': self.config.n_generation_trials,
            'total_iterations': self.config.total_iterations,
            'n_inner': 1,
            'iteration_num': self._iteration,
            'target_character': target_character,
            'pe': self.config.pe,
            'branch_factor': self.config.branch_factor,
            'feedback_path': self.previous_feedback_path,
            'map_width': self.config.map_width,
            'map_height': self.config.map_width,
            'feature': self.config.reward_feature,
            'available_tiles': get_available_tiles(self.config.problem),
            'prev_eval_result': prev_eval_result,
            'auxiliary_prompt_path': auxiliary_prompt_path,
            'n_codegen_trials': self.config.n_codegen_trials,
            'n_codefix_trials': self.config.n_codefix_trials,
            'task': self.config.task,
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
        origin_path = path.join(dirname(__file__), 'pcgrllm', 'bypass_reward', reward_filename)
        target_path = path.join(self.reward_functions_dir, reward_filename)

        self.logging(f"Copying reward function to the experiment directory: {origin_path} -> {target_path}", logging.WARNING)
        shutil.copy(origin_path, target_path)

        return target_path

    def bypass_feedback(self, iteration_num: int):
        # copy the reward function

        os.makedirs(self.feedback_dir, exist_ok=True)

        reward_filename = f'{basename(self.config.bypass_feedback_path)}.txt'
        origin_path = path.join(dirname(__file__), 'pcgrllm', 'bypass_feedback', reward_filename)
        target_path = path.join(self.feedback_dir, reward_filename)

        self.logging(f"Copying feedback to the experiment directory: {origin_path} -> {target_path}", logging.WARNING)
        shutil.copy(origin_path, target_path)

        return target_path

    def bypass_train(self):
        # copy the directory

        target_path = path.join(self.config.exp_dir, f'iteration_{self._iteration}')
        os.makedirs(target_path, exist_ok=True)

        origin_path = path.join(dirname(__file__), 'pcgrllm', 'bypass_train', self.config.bypass_train_path)
        self.logging(f"Copying train to the experiment directory: {origin_path} -> {target_path}", logging.WARNING)
        shutil.copytree(origin_path, target_path, dirs_exist_ok=True)

        # replace the name of *.py file into 'reward_outer_{self._iteration}_inner_1.py'
        reward_file = glob(path.join(target_path, '*.py'))[0]
        new_reward_file = path.join(target_path, f"reward_outer_{self._iteration}_inner_1.py")

        self.logging(f"Renaming the reward function file: {reward_file} -> {new_reward_file}", logging.WARNING)
        os.rename(reward_file, new_reward_file)

        return target_path

    def train_pcgrl(self):
        """Runs the mlagents-learn command with the given parameters."""

        config = copy.deepcopy(self.config)
        config.exp_dir = path.join(config.exp_dir, f'iteration_{self._iteration}')
        config.initialize = False
        config.current_iteration = self._iteration
        # config.initialize_wandb = False # Disable wandb in train.py

        media_dir = path.join(config.exp_dir, 'train')
        os.makedirs(media_dir, exist_ok=True)
        config._vid_dir = os.path.join(media_dir, 'videos')
        config._img_dir = os.path.join(media_dir, 'images')
        config._numpy_dir = os.path.join(media_dir, 'numpy')

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


    def rollout_pcgrl(self, iteration_run_id) -> str:
        from rollout import main_rollout as run_rollout

        config = copy.deepcopy(self.config)
        config.initialize = False
        config.exp_dir = path.join(config.exp_dir, 'iteration_' + str(iteration_run_id))
        config.random_agent = False

        media_dir = path.join(config.exp_dir, 'inference')
        os.makedirs(media_dir, exist_ok=True)

        config._vid_dir = os.path.join(media_dir, 'videos')
        config._img_dir = os.path.join(media_dir, 'images')
        config._numpy_dir = os.path.join(media_dir, 'numpy')
        config.n_envs = self.config.n_samples
        config.n_eps = 1

        run_rollout(config)

        return media_dir

    def bypass_rollout(self, iteration_run_id) -> str:
        config = copy.deepcopy(self.config)
        config.exp_dir = path.join(config.exp_dir, 'iteration_' + str(iteration_run_id))
        media_dir = path.join(config.exp_dir, 'inference')

        return media_dir

    def differ_type_feedback(self, iteration_num: int):
        # copy the reward function

        os.makedirs(self.feedback_dir, exist_ok=True)
        if self.config.feedback_type == 'no':
            reward_filename = f'{basename(self.config.feedback_type)}.txt'
        else:
            reward_filename = f'{basename(self.config.feedback_type + "_" + self.config.task + "_feedback")}.txt'

        origin_path = path.join(dirname(__file__), 'pcgrllm', 'bypass_feedback', reward_filename)
        target_path = path.join(self.feedback_dir, reward_filename)

        self.logging(f"Copying feedback to the experiment directory: {origin_path} -> {target_path}", logging.WARNING)
        shutil.copy(origin_path, target_path)

        return target_path
    # 파일 분석
    def analyze_output(self, iteration_num: int) -> None:

        if self.config.task == TaskType.Alphabet:
            condition_prompt = f'Make a level looks like "{self.config.target_character}"'
        elif self.config.task == TaskType.Scenario:
            condition_prompt = ScenarioPreset().scenarios.get(self.config.target_character)
        else:
            raise ValueError(f"Invalid task type: {self.config.task}")

        args_dict = {
            'exp_path': self.config.exp_dir,
            'condition_prompt': condition_prompt,
            'input_type': self.config.feedback_input_type,
            'gpt_model': self.config.gpt_model,
            'reward_function': self._current_reward_function_filename,
            'available_tiles': get_available_tiles(self.config.problem),
            'iteration': self._iteration,
            'task': self.config.task
        }

        feedback = generate_feedback(self.config, args_dict)
        # red

        tgt_dir = path.join(self.config.exp_dir, 'iteration_' + str(iteration_num))
        # save feedback text to iteratio, dir
        feedback_path = path.join(tgt_dir, f'feedback.txt')
        # copy feedback using shutil
        shutil.copy(feedback, feedback_path)

        return feedback_path

    def run_evaluation(self):
        exp_dir = path.join(self.config.exp_dir, f'iteration_{self._iteration}')
        iteration = self.storage.get_iteration(self._iteration)

        if self.config.task == TaskType.Alphabet:
            result = self._run_alphabet_evaluation(iteration=iteration, target_character=self.config.target_character)
        elif self.config.task == TaskType.Scenario:
            result = self._run_scenario_evaluation(iteration=iteration, scenario_num=self.config.target_character)
        else:
            raise ValueError(f"Invalid task type: {self.config.task}")

        # Write the fitness to the file
        if self.config.random_fitness:
            result = EvaluationResult(task=self.task).sample() # TODO Check before running experiment

        # save to the iteration file
        result_path = path.join(exp_dir, 'evaluation.json')
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f)

        if self.config.pe in ['tot', 'got']:
            self.logging(f"Node evaluation score: {result.total}", level=logging.INFO)
            self.graph_manager.update(self.current_node, fitness=result.total, refer_ids=self.auxiliary_iter_nums)


    def _run_alphabet_evaluation(self, iteration: Iteration, target_character: str) -> EvaluationResult:
        exp_dir = path.join(self.config.exp_dir, f'iteration_{self._iteration}')

        if self.config.evaluator == 'vit':
            evaluator = ViTEvaluator(task=self.config.task, logger=self.logger)
        elif self.config.evaluator == 'hr':
            evaluator = HeuristicEvaluator(task=self.config.task, logger=self.logger)
        elif self.config.evaluator == 'llm':
            evaluator = LLMEvaluator(task=self.config.task, logger=self.logger,
                                     gpt_model=self.config.gpt_model, seed=self.config.seed,
                                     n_generation_trials=self.config.n_generation_trials)
        else:
            raise ValueError(f"Invalid evaluator type: {self.config.evaluator}")

        result = evaluator.run(iteration=iteration, target_character=target_character)

        if self.config.evaluator == 'vit':
            log_evaluation_result(logger=self.logger, result=result, iteration=self._iteration, evaluator_type=None)
        else:
            log_evaluation_result(logger=self.logger, result=result, iteration=self._iteration, evaluator_type=self.config.evaluator)

            # Get the evaluation result
            vit_evaluator = ViTEvaluator(task=self.config.task, logger=self.logger)
            vit_result = vit_evaluator.run(iteration=iteration, target_character=self.config.target_character)

            # Save the evaluation result to the iteration file
            result_path = path.join(exp_dir, 'evaluation.vit.json')
            with open(result_path, 'w') as f:
                json.dump(vit_result.to_dict(), f)

            # Log the evaluation result
            log_evaluation_result(logger=self.logger, result=vit_result, iteration=self._iteration, evaluator_type=None)
            self.logging(f"ViT Result: {vit_result}", level=logging.INFO)

        return result

    def _run_scenario_evaluation(self, iteration: Iteration, scenario_num: str) -> EvaluationResult:

        exp_dir = path.join(self.config.exp_dir, f'iteration_{self._iteration}')

        if self.config.evaluator == 'hr':
            evaluator = SolutionEvaluator(task=self.config.task, logger=self.logger)
        elif self.config.evaluator == 'llm':
            evaluator = LLMEvaluator(task=self.config.task, logger=self.logger,
                                     gpt_model=self.config.gpt_model, seed=self.config.seed,
                                     n_generation_trials=self.config.n_generation_trials)
        else:
            raise ValueError(f"Invalid evaluator type: {self.config.evaluator}")

        result = evaluator.run(iteration=iteration, target_character=scenario_num)

        if self.config.evaluator == 'hr':
            log_evaluation_result(logger=self.logger, result=result, iteration=self._iteration, evaluator_type=None)
        else:
            log_evaluation_result(logger=self.logger, result=result, iteration=self._iteration, evaluator_type=self.config.evaluator)

            # Get the evaluation result
            sol_evaluator = SolutionEvaluator(task=self.config.task, logger=self.logger)
            sol_result = sol_evaluator.run(iteration=iteration, target_character=self.config.target_character)

            # Save the evaluation result to the iteration file
            result_path = path.join(exp_dir, 'evaluation.vit.json')
            with open(result_path, 'w') as f:
                json.dump(sol_result.to_dict(), f)

            # Log the evaluation result
            log_evaluation_result(logger=self.logger, result=sol_result, iteration=self._iteration, evaluator_type=None)
            self.logging(f"Solution Result: {sol_result}", level=logging.INFO)

        return result


    def save_state(self):
        """Saves all instance variables to a YAML file, excluding specified keys."""
        exclude_keys = ['config', 'storage', 'graph_manager', 'logger']

        # Filter out excluded keys
        data_to_serialize = {key: value for key, value in self.__dict__.items() if key not in exclude_keys}

        if 'current_node' in data_to_serialize:
            data_to_serialize['current_node'] = data_to_serialize['current_node'].to_dict()

        # Save to YAML
        with open(path.join(self._experiment_path, 'state.json'), 'w') as file:
            json.dump(data_to_serialize, file)

    def save_previous_reward(self):
        with open(path.join(self.config.exp_dir, f"iteration_{self._iteration}", "previous_reward_function.json"), 'w') as file:
            yaml.dump({"previous_reward_function": self.previous_reward_function_path}, file, indent=4)

    def save_best_reward(self):
        with open(path.join(self.config.exp_dir, "best_reward_function.json"), 'w') as file:
            yaml.dump({"best_reward_function": self.previous_reward_function_path}, file, indent=4)

    def load_state(self):
        json_path = path.join(self._experiment_path, 'state.json')

        if path.exists(json_path):

            with open(json_path, 'r') as file:
                state = json.load(file)

            self.logging(f"Loading state from{json_path}:\n{state}", level=logging.INFO)

            for key, value in state.items():
                setattr(self, key, value)

            if 'current_node' in state:
                self.current_node = NodeInfo.from_dict(state['current_node'])

        if self._stage == Stage.Done:
            self._stage = Stage.FinishIteration
            self.logging("The experiment has already finished, but check if there is any remaining work to do.", level=logging.INFO)

    def get_evaluation_result(self, iteration_num: int) -> Optional[EvaluationResult]:
        """Returns the evaluation result for the given iteration number."""
        iteration = self.storage.get_iteration(iteration_num)
        return iteration.get_evaluation()

    def run(self):

        self.logging("Running experiment", level=logging.INFO)

        start_wandb(config=self.config)

        while not self._stage is Stage.Done:

            self.logging(f"Current stage: {self._stage}", level=logging.INFO)


            if self._stage == Stage.StartIteration:

                if self.config.pe in ['tot', 'got']:
                    self.current_node = self.graph_manager.expand_node(self._iteration)
                    iteration = self.storage.get_iteration(self.current_node.parent_id)

                    if iteration is not None:
                        self.previous_reward_function_path = iteration.get_reward_function_path()
                        self.previous_feedback_path = iteration.get_feedback_path()

                if self.config.fewshot is True and self._iteration == 1:
                    fewshot_reward = path.join(dirname(__file__), 'pcgrllm', 'bypass_reward', 'fewshot', f'shape_{self.config.target_character.lower()}.py')
                    self.logging(f"Fewshot reward function: {fewshot_reward}", level=logging.INFO)
                    self.previous_reward_function_path = fewshot_reward

                self._stage = Stage.RewardGeneration

            elif self._stage == Stage.RewardGeneration:

                if self.config.bypass_reward_path is not None:
                    reward_generation_fn = self.bypass_reward_function
                else:
                    reward_generation_fn = self.generate_reward_function

                self._current_reward_function_filename = reward_generation_fn()

                if self._current_reward_function_filename is False:
                    self.exit("Reward function generation failed. Exiting.", code=1)
                else:

                    reward_function_dir = path.join(self.reward_functions_dir, f'reward_outer_{self._iteration}_inner_1')
                    log_reward_generation_data(logger=self.logger, target_path=reward_function_dir, iteration=self._iteration)
                    self._stage = Stage.TrainPCGRL

            elif self._stage == Stage.TrainPCGRL:
                # Run ML-Agents
                if self.config.bypass_train_path is not None:
                    train_fn = self.bypass_train
                else:
                    train_fn = self.train_pcgrl

                train_fn()

                self._stage = Stage.RolloutPCGRL

            elif self._stage == Stage.RolloutPCGRL:
                # Collect results

                if self.config.bypass_train_path is not None:
                    rollout_pcgrl_fn = self.bypass_rollout
                else:
                    rollout_pcgrl_fn = self.rollout_pcgrl

                output_dir = rollout_pcgrl_fn(self._iteration)

                log_rollout_data(logger=self.logger, target_path=output_dir, iteration=self._iteration)


                self._stage = Stage.Evaluation

            elif self._stage == Stage.Evaluation:

                self.run_evaluation()

                if self._iteration >= self.config.total_iterations:
                    self._stage = Stage.FinishIteration
                else:
                    self._stage = Stage.Analysis

            elif self._stage == Stage.Analysis:
                # Analyze results

                if self.config.bypass_feedback_path is not None:
                    feedback_generation_fn = self.bypass_feedback
                elif self.config.feedback_type != 'default':
                    feedback_generation_fn = self.differ_type_feedback
                else:
                    feedback_generation_fn = self.analyze_output

                self._current_feedback_path = feedback_generation_fn(self._iteration)

                log_feedback_data(logger=self.logger, target_path=path.join(dirname(self._current_feedback_path), 'feedback'), iteration=self._iteration)

                self._stage = Stage.FinishIteration

            elif self._stage == Stage.FinishIteration:
                self.previous_reward_function_path = self._current_reward_function_filename
                self.previous_feedback_path = self._current_feedback_path

                if self.config.pe in ['tot', 'got']:
                    self.storage.get_iteration(self._iteration).set_node(self.current_node)
                    self.logging('Tree Status:\n' + self.graph_manager.print_tree(iteration_marker=self._iteration, best_marker=True), level=logging.INFO)

                if self._iteration >= self.config.total_iterations:
                    self.save_best_reward()
                    self._stage = Stage.Done
                else:
                    self._iteration += 1
                    self._stage = Stage.StartIteration

            self.save_state()

        finish_wandb()
        self.logging("Experiment finished.")

    def exit(self, message: str, code: int = 1):
        self.logging(message, level=logging.ERROR)

        if code != 0:
            raise SystemExit(message)
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
    # if the problem is scenario, set the problem to dungeon3
    init_config(config)

    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    main()

