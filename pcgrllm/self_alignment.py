import copy
import datetime
import math
import os, re, ast, sys, time, argparse, json, pickle
import pprint
import shutil
import traceback
import warnings
from copy import deepcopy

from os.path import abspath, basename
import logging

import numpy as np

from conf.config import TrainLLMConfig, Config

from pcgrllm.utils.exceptions import RewardExecutionException, RewardParsingException
from pcgrllm.utils.path_utils import init_config
from pcgrllm.validate_reward import run_validate, read_file

logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)

from pcgrllm.llm_client.llm import ChatContext, UnifiedLLMClient
from pcgrllm.llm_client.utils import *

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
# set logging level to DEBUG
logging.basicConfig(level=getattr(logging, log_level, logging.DEBUG))
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.DEBUG))

PE_DIR = 'pe'
FEATURE_DIR = 'feature'
TASK_DIR = 'task'


class SelfAlignment:
    def __init__(self, config: dict):
        self.config = config

        self.api_key = config.get('api_key')
        self.shared_storage_path = config.get('shared_storage_path', '.')
        self.postfix = config.get('postfix', time.strftime("%Y-%m-%d-%H-%M-%S"))
        self.reward_functions_dir = config.get('reward_functions_dir', 'RewardFunctions')
        self.gpt_model = config.get('gpt_model', 'gpt-3.5-turbo')
        self.gpt_max_token = config.get('gpt_max_token', 1024)
        self.verbose = config.get('verbose', False)

        self.n_inner = config.get('n_inner', 2)
        self.n_outer = config.get('n_outer', 3)

        self._current_trial = 0

        self.n_codegen_trials = config.get('n_codegen_trials', 1)
        self.n_codefix_trials = config.get('n_codefix_trials', 3)

        self.iteration_num = config.get('iteration_num', 1)

        self.reward_function_path = path.join(self.shared_storage_path, self.reward_functions_dir,
                                              (str(self.postfix) + '_inner_' + str(self.n_inner)))

        self.file_path = path.join(self.shared_storage_path, 'prompt', self.config['task'])
        self.user_prompt = file_to_string(path.join(self.file_path, "alignment_user.txt"))


        self.previous_reward_function_path = config.get('previous_reward_function')
        self.previous_reward_function = file_to_string(self.previous_reward_function_path)

        os.makedirs(self.reward_function_path, exist_ok=True)

        self.initial_reward_function = config.get('initial_reward_function', None)

        if self.initial_reward_function is not None:
            self._prepare_initial_reward_function()

        self.logging(f'Reward generation arguments:\n{pprint.pformat(config, indent=4)}', logging.INFO)

        os.makedirs(self.reward_function_path, exist_ok=True)


    def logging(self, message, level=logging.DEBUG):
        info_dict = {
            'outer_loop': self.iteration_num if hasattr(self, 'iteration_num') else -1,
            # 'innen_innerr_loop': self.current_inner,
            'n_inner': self.n_inner,
            'codegen_total': self.n_codegen_trials,
            'codegen_trial': self._curr_codegen_trial if hasattr(self, '_curr_codegen_trial') else -1,
            'fix_total': self.n_codefix_trials,
            'fix_trial': self._curr_codefix_trial if hasattr(self, '_curr_codefix_trial') else -1,
        }

        # Define the prefix format
        prefix = '[iter: {outer_loop}, codegen: {codegen_trial}/{codegen_total}, fixing: {fix_trial}/{fix_total}]'.format(
            **info_dict)

        # Split the message by line breaks and log each line with the prefix
        message = str(message)
        for line in message.splitlines():
            formatted_message = f'{prefix} {line}'
            logger.setLevel(logging.DEBUG)
            logger.log(level, formatted_message)

    def _prepare_initial_reward_function(self):
        # Copy the initial reward function to the reward function path
        reward_file_name = basename(self.initial_reward_function)

        initial_reward_function_path = path.join(self.reward_function_path, reward_file_name)
        # Copy
        shutil.copy(self.initial_reward_function, initial_reward_function_path)

        self.generating_function_path = initial_reward_function_path
        self.previous_reward_function = file_to_string(self.generating_function_path)
        self.logging(
            f"Copied the initial reward function to the reward function path: {self.initial_reward_function} -> {initial_reward_function_path}",
            logging.INFO)

    def start_chat(self, model, messages, max_tokens, log_dict=None, log_key='first_user', passthrough_response=None,
                   verbose=False, seed=42, n_response=1):
        try:
            if passthrough_response is None:
                if verbose:
                    self.logging("Sending the request: ", messages)

                client = UnifiedLLMClient()
                ctx = ChatContext()

                responses = client.call_model(ctx, messages, model=model, seed=seed, n_response=n_response)

                response = responses[0][0]
                context = responses[0][1]

                if verbose:
                    self.logging("Received the response: ", response)
            else:
                try:
                    response = file_to_string(passthrough_response)
                except FileNotFoundError as e:
                    self.logging(logging.CRITICAL, "File not found: {passthrough_response}\n", e)
                    raise Exception("Raise Error")

            if log_dict is not None:
                log_dict[log_key] = dict(request=messages, response=response)

        except KeyboardInterrupt:
            raise Exception("Keyboard Interrupt while using the OpenAI API")


        return response, context

    def run(self, return_error: bool = False):

        is_success = False

        reward_function_name = f"{self.postfix}_inner_{self.n_inner}"

        for codegen_trial in range(1, self.n_codegen_trials + 1):
            self._curr_codegen_trial = codegen_trial

            fn_codegen_name = f"{reward_function_name}_trial_{codegen_trial}"
            fn_codegen_dir = path.join(self.reward_function_path, fn_codegen_name)
            os.makedirs(fn_codegen_dir, exist_ok=True)

            generating_function_path = None
            generating_function_error = None

            if hasattr(self, 'generating_function_path'):
                generating_function_path = self.generating_function_path
                self.previous_reward_function_path = self.generating_function_path
                del self.generating_function_path
                self.logging(f"Using the initial reward function: {generating_function_path}", logging.INFO)
            if generating_function_path is None and self.previous_reward_function_path is not None:
                generating_function_path = self.previous_reward_function_path



            self.logging(f"Generating reward function: {generating_function_path}", logging.DEBUG)


            for codefix_trial in range(0, self.n_codefix_trials + 1):
                self._curr_codefix_trial = codefix_trial

                fn_codefix_name = f"{fn_codegen_name}_fix_{codefix_trial}"

                self.logging(f"Generating reward function: {fn_codegen_name}", logging.INFO)

                self.logging(
                    f'Calling the zero-shot generation function. (len(error): {len(generating_function_error) if generating_function_error else 0})')
                generating_function_path = self.first_user_response(base_name=fn_codefix_name,
                                                                    base_directory=fn_codegen_dir,
                                                                    generating_function_path=generating_function_path,
                                                                    generating_function_error=generating_function_error,
                                                                    trial=codegen_trial)
                self.logging(f'Called the first_user_response function')

                # copy file generating_function_path to fn_codegen_name.py
                shutil.copy(generating_function_path, path.join(self.reward_function_path, f"{fn_codegen_name}.py"))

                error_message = None

                try:
                    self.logging(f"Parsing the reward function: {generating_function_path}", logging.DEBUG)

                    reward_function = self.parse_reward_function(generating_function_path)

                    # Overwrite the reward function
                    with open(generating_function_path, 'w') as f:
                        f.write(reward_function)

                    if hasattr(self, '_execution_config'):
                        self.logging(f"Validating the reward function: {generating_function_path}", logging.DEBUG)
                        self.execute_reward_function(generating_function_path)
                    else:
                        self.logging(f'Execution config is not set. Skipping the validation.', logging.WARNING)

                    is_success = True

                except Exception as e:
                    self.logging(f"Failed to generating the reward function: {generating_function_path}", logging.INFO)
                    self.logging(str(e), logging.DEBUG)
                    error_message = str(e)
                    is_success = False

                if is_success:
                    self.previous_reward_function = file_to_string(generating_function_path)
                    self.previous_reward_function_path = generating_function_path

                    # copy to reward file

                    break
                else:
                    generating_function_error = error_message

            if is_success:
                break
            else:
                self.logging(f"Failed to generate the reward function: {generating_function_path}", logging.WARNING)

        if is_success:
            reward_function_string = file_to_string(generating_function_path)
            reward_function_file_path = path.join(self.reward_function_path, f"{reward_function_name}.py")

            with open(reward_function_file_path, 'w') as f:
                self.logging(f"Saving the reward function to the file: {reward_function_file_path}", logging.INFO)
                f.write(reward_function_string)

            return reward_function_file_path
        else:
            self.logging(f"Failed to generate the reward function: {reward_function_name}", logging.WARNING)

            if return_error:
                return False, generating_function_error
            else:
                return False

    def set_execution_config(self, config: Config):
        self.logging(f"Setting the execution config: {config}", logging.INFO)
        self._execution_config = config


    def save_data(self, response: str, context: list, messages: list,
                  base_directory: str = '',
                  base_name: str = 'reward', branch: int = None):

        self.logging(context, logging.INFO)
        self.logging(response, logging.DEBUG)

        if branch is None:
            file_name = f"{base_name}"
        else:
            file_name = f"{base_name}_branch_{branch}"

        response_file_path = path.join(base_directory, f"{base_name}.response.pkl")
        with open(response_file_path, 'wb') as f:
            pickle.dump(response, f)

        context_file_path = path.join(base_directory, f"{base_name}.context.pkl")
        with open(context_file_path, 'wb') as f:
            pickle.dump(context, f)

        parsed_reward_function = parse_reward_function(response)

        log_dict = {
            'request': messages,
            'response': response,
        }

        # Save reward function to .py
        reward_file_path = path.join(base_directory, f"{file_name}.py")
        with open(reward_file_path, 'w') as f:
            f.write(parsed_reward_function)

        # Save the log to .json file
        log_file_path = path.join(base_directory, f"{file_name}.json")
        with open(log_file_path, 'w') as f:
            json.dump(log_dict, f, indent=4)

        return reward_file_path

    def get_reward_simulation_prompt(self, reward_function_path: str):
        _, rewards = self.execute_reward_function(reward_function_path)

        rewards = np.array(rewards).reshape(-1)

        mean = np.mean(rewards)
        std = np.std(rewards)
        zero_perc = np.sum(rewards == 0) / len(rewards) * 100

        return f"Mean: {mean:.6f}, Std: {std:.6f}, Zero Value Percent: {zero_perc:.4f}%"

    def first_user_response(self,
                            base_name: str = 'reward',
                            base_directory: str = None,
                            generating_function_path: str = None,
                            generating_function_error: str = None,
                            trial=1):

        initial_user = copy.deepcopy(self.user_prompt)

        rollout_result = self.get_reward_simulation_prompt(self.previous_reward_function_path)


        if generating_function_error:
            reward_code = file_to_string(generating_function_path)

            error_prompt = """
            ## Reward Code
            Here is the previous reward function that you have generated. However, this code has an error. Please fix the error and generate the reward function again.

            Error Message:
            {error_message}

            """.format(reward_code_string=reward_code, error_message=generating_function_error)

            initial_user = initial_user.format(
                reward_function=self.previous_reward_function,
                rollout_result=rollout_result,
                error_prompt=error_prompt,
            )
        else:
            initial_user = initial_user.format(
                reward_function=self.previous_reward_function,
                rollout_result=rollout_result,
                error_prompt='',
            )

        # 피드백 받는 부분 작성 필요함

        messages = [
            {"role": "user", "content": initial_user}
        ]

        self.logging(f'Input to the reward generation model:\n{json.dumps(messages, indent=2)}', logging.DEBUG)

        response, context = self.start_chat(self.gpt_model, messages, self.gpt_max_token, seed=trial)

        self.logging(context, logging.INFO)
        self.logging(response, logging.DEBUG)

        response_file_path = path.join(base_directory, f"{base_name}.response.pkl")
        with open(response_file_path, 'wb') as f:
            pickle.dump(response, f)

        context_file_path = path.join(base_directory, f"{base_name}.context.pkl")
        with open(context_file_path, 'wb') as f:
            pickle.dump(context, f)

        reward_file_path = self.save_data(response=response,
                                          context=context,
                                          base_directory=base_directory,
                                          base_name=base_name,
                                          messages=messages)

        return reward_file_path

    def parse_reward_function(self, reward_function_path: str) -> str:
        # Return the parsed reward function
        try:
            code = parse_reward_function(file_to_string(reward_function_path))
        except Exception as e:
            raise RewardParsingException(file_to_string(reward_function_path), e)

        return code

    def execute_reward_function(self, reward_function_path: str) -> bool:
        config = deepcopy(self._execution_config)
        config.reward_function_path = reward_function_path

        try:
            result, rewards = run_validate(config, return_reward=True, length=math.pow(config.map_width, 2) * 3)
        except Exception as e:
            code = read_file(reward_function_path)

            # get traceback from e
            tb_list = traceback.format_exception(type(e), e, e.__traceback__)

            string_lines = [line.strip() for line in tb_list if "File \"<string>\"" in line]
            filtered_traceback = ''.join(string_lines).strip()

            raise RewardExecutionException(code, f'{filtered_traceback}\n{e}')

        return result, rewards


def self_alignment(config: Config, generate_args: dict, return_error: bool = False):
    aligner = SelfAlignment(generate_args)
    aligner.set_execution_config(config)
    return aligner.run(return_error=return_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--shared_storage_path', type=str, default=abspath('.'))
    parser.add_argument('--postfix', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    parser.add_argument('--reward_functions_dir', type=str, default='reward_functions')
    parser.add_argument('--gpt_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--gpt_max_token', type=int, default=4096)
    parser.add_argument('--current_inner', type=int, default=1)
    parser.add_argument('--task', type=str, default='alphabet')
    parser.add_argument('--n_codegen_trials', type=int, default=3)
    parser.add_argument('--n_codefix_trials', type=int, default=3)
    parser.add_argument('--iteration_num', type=int, default=1)
    parser.add_argument('--previous_reward_function', type=str, default=None, required=True)

    args = parser.parse_args()

    args = vars(args)

    train_config = TrainLLMConfig()
    init_config(train_config)

    reward_generator = SelfAlignment(args)
    reward_generator.set_execution_config(train_config)

    reward_generator.run()