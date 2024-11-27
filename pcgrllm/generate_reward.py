import copy
import datetime
import os, re, ast, sys, time, argparse, json, pickle
import pprint
import shutil
import traceback
import warnings
from copy import deepcopy

import numpy as np

from os.path import abspath, basename
import logging
# from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


from conf.config import TrainLLMConfig, Config
from envs.probs.binary import BinaryTiles

from envs.probs.dungeon2 import Dungeon2Tiles
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


class RewardGenerator:
    def __init__(self, config: dict):
        self.config = config

        self.api_key = config.get('api_key')
        self.shared_storage_path = config.get('shared_storage_path', '.')
        self.postfix = config.get('postfix', time.strftime("%Y-%m-%d-%H-%M-%S"))
        self.reward_functions_dir = config.get('reward_functions_dir', 'RewardFunctions')
        self.gpt_model = config.get('gpt_model', 'gpt-3.5-turbo')
        self.gpt_max_token = config.get('gpt_max_token', 1024)
        self.verbose = config.get('verbose', False)
        self.n_inner = 1  # Legacy

        self.current_inner = config.get('current_inner', 1)

        self.n_outer = config.get('n_outer', 3)
        self._current_trial = 0

        self.n_codegen_trials = config.get('n_codegen_trials', 1)
        self.n_codefix_trials = config.get('n_codefix_trials', 3)

        self.iteration_num = config.get('iteration_num', 1)
        self.branch_factor = config.get('branch_factor', None)

        self.arbitrary_dataset = config.get('arbitrary_dataset', 'arbitrary_dataset.txt')
        self.file_path = path.join(self.shared_storage_path, 'prompt')

        self.prev_eval_result = config.get('prev_eval_result', None)
        self.auxiliary_prompt_path = config.get('auxiliary_prompt_path', None)

        self.example_path = path.join(self.shared_storage_path, 'example')
        self.current_state_path = path.abspath(path.join(self.shared_storage_path, 'example', 'testState.json'))
        self.reward_function_path = path.join(self.shared_storage_path, self.reward_functions_dir,
                                              (str(self.postfix) + '_inner_' + str(self.n_inner)))
        self.initial_system = file_to_string(path.join(self.file_path, "system.txt"))
        self.initial_user = file_to_string(path.join(self.file_path, "initial_user.txt"))

        self.jax_code_tips_prompt = file_to_string(path.join(self.file_path, "jax_code_tips.txt"))
        self.reward_code_tips_prompt = file_to_string(path.join(self.file_path, "reward_code_tips.txt"))

        self.task_description = file_to_string(path.join(self.file_path, "task_description.txt"))
        # self.second_user = file_to_string(path.join(self.file_path, "second_user.txt"))

        self.reward_function_inputs_template = file_to_string(path.join(self.file_path, "reward_function_inputs.txt"))

        self.reward_template = file_to_string(path.join(self.file_path, "compute_reward_example.py"))

        self.pe = config.get('pe')
        self.feature = config.get('feature')

        if self.feature is not None:
            self.feature = self.feature.split('+')

        self.task = config.get('task')
        self.available_tiles = config.get('available_tiles')

        # previous 나중에 변경하기
        default_reward = path.join(self.file_path, "compute_reward_example.py")

        self.previous_reward_function_path = config.get('previous_reward_function', None)
        if self.previous_reward_function_path is None:
            self.previous_reward_function_path = default_reward
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
                if n_response != 1:
                    temperature = 0.5
                else:
                    temperature = 0
                responses = client.call_model(ctx, messages, model=model, seed=seed, n_response=n_response,
                                              temperature=temperature)
                if n_response != 1:
                    generated_responses = [response[0] for response in responses]
                    generated_contexts = [response[1] for response in responses]
                    index = 1
                    # ndex = self.response_cluster(generated_responses, n_clusters=3)
                else:
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

        if n_response != 1:
            return generated_responses, generated_contexts, index
        else:
            return response, context

    def run(self, return_error: bool = False):

        is_success = False

        reward_function_name = f"{self.postfix}_inner_{self.current_inner}"

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

            # 여기서 수정 하는것만 짜자

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

    # def response_cluster(self, responses, n_clusters=2):
    #     # set environment variable for
    #
    #     os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    #
    #     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #     embeddings = model.encode(responses)
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    #     labels = kmeans.fit_predict(embeddings)
    #
    #     clusters = {}
    #     for i, label in enumerate(labels):
    #         if label not in clusters:
    #             clusters[label] = []
    #         clusters[label].append(i)
    #
    #     largest_cluster = max(clusters.values(), key=len)
    #     largest_cluster_embeddings = embeddings[largest_cluster]
    #
    #     cluster_center = np.mean(largest_cluster_embeddings, axis=0)
    #     index = min(largest_cluster, key=lambda i: np.linalg.norm(embeddings[i] - cluster_center))
    #     return index

    def set_execution_config(self, config: Config):
        self.logging(f"Setting the execution config: {config}", logging.INFO)
        self._execution_config = config

    def get_feedback_prompt(self):
        feedback_prompt = file_to_string(self.config['feedback_path'])
        return feedback_prompt

    def get_input_prompt(self):

        prompt = copy.deepcopy(self.reward_function_inputs_template)

        if 'array' in self.feature:
            prompt = prompt.replace('{array_feature_prompt}', self._get_array_feature())
        else:
            prompt = prompt.replace('{array_feature_prompt}', '(array features not available)')
            prompt = prompt.replace('prev_array', 'unused1')
            prompt = prompt.replace('curr_array', 'unused2')

        if 'stats' in self.feature:
            prompt = prompt.replace('{stats_feature_prompt}', self._get_stats_feature())
        else:
            prompt = prompt.replace('{stats_feature_prompt}', '(stats features not available)')
            prompt = prompt.replace('prev_stats', 'unused3')
            prompt = prompt.replace('curr_stats', 'unused4')
        return prompt

    def _get_array_feature(self):
        prompt = self.get_feature_prompt('array')

        # level_shape_str = f"({self.config['map_height']}, {self.config['map_width']})"

        available_tiles = set(self.available_tiles) | {BinaryTiles.EMPTY, BinaryTiles.WALL}

        # Filter the enum members based on available_tiles
        tile_enum = ', '.join(f"{tile.name} = {tile.value}" for tile in Dungeon2Tiles if tile in available_tiles)

        # Format the prompt with values
        return prompt.format(
            # array_shape=level_shape_str,
            tile_enum=tile_enum
        )

        return prompt

    def _get_stats_feature(self):
        prompt = self.get_feature_prompt('stats')
        return prompt

    def get_initial_user(self):
        initial_user = copy.deepcopy(self.initial_user)

        # get the prompt file from the task dir
        task_file = path.join(self.file_path, TASK_DIR, f"{self.task}.txt")
        task_prompt = file_to_string(task_file)

        task_prompt.format(target_character=self.config['target_character'])

        initial_user = initial_user.replace('{task}', task_prompt)

        return initial_user

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

    def first_user_response(self,
                            base_name: str = 'reward',
                            base_directory: str = None,
                            generating_function_path: str = None,
                            generating_function_error: str = None,
                            trial=1):

        initial_system = self.initial_system.format(
            i='{i}',
            reward_signature=self.reward_template,
        )

        # Add jax code tips prompt
        initial_system += self.jax_code_tips_prompt
        initial_system += '\n'
        initial_system += self.reward_code_tips_prompt

        initial_user = self.get_initial_user()

        reward_function_inputs = self.get_input_prompt()

        if generating_function_error:
            reward_code = file_to_string(generating_function_path)

            sample_code = """
            ## Reward Code
            Here is the previous reward function that you have generated. However, this code has an error. Please fix the error and generate the reward function again.
            ```python
            {reward_code_string}
            ```
            Error Message:
            {error_message}

            """.format(reward_code_string=reward_code, error_message=generating_function_error)

            initial_user = initial_user.format(
                few_shot_code_string=sample_code,
                reward_function_inputs=reward_function_inputs,
                target_character=self.config['target_character'],
                thought_tips=self.get_pe_prompt(self.pe),
            )


        elif self.config['feedback_path'] is not None:  # Feedback available

            reward_code = file_to_string(generating_function_path)

            sample_code = """
               ## Previous Reward Code
               Here is the previous reward function that you have generated. However, this code has an error. Please fix the error and generate the reward function again.
               ```python
               {reward_code_string}
               ```

               ### Feedback:
               {feedback}
                Please update the reward function based on the feedback and generate the reward function again.
               """.format(reward_code_string=reward_code, feedback=self.get_feedback_prompt())

            initial_user = initial_user.format(
                few_shot_code_string=sample_code,
                reward_function_inputs=reward_function_inputs,
                target_character=self.config['target_character'],
                thought_tips=self.get_pe_prompt(self.pe),
            )

        else:
            sample_code = """
            ## Example Reward Code
            ```python
            {sample_reward_code}
            ```
            """.format(sample_reward_code=self.previous_reward_function)

            # if the every first iteration, use the io prompt
            if self.iteration_num == 1:
                pe_prompt = self.get_pe_prompt('io')
            else:
                pe_prompt = self.get_pe_prompt(self.pe)

            initial_user = initial_user.format(
                target_character=self.config['target_character'],
                few_shot_code_string=sample_code,
                reward_function_inputs=reward_function_inputs,
                thought_tips=pe_prompt,
            )

        # 피드백 받는 부분 작성 필요함

        messages = [
            {"role": "system", "content": initial_system},
            {"role": "user", "content": initial_user}
        ]

        self.logging(f'Input to the reward generation model:\n{json.dumps(messages, indent=2)}', logging.DEBUG)

        if self.pe == 'cotsc':
            responses, contexts, index = self.start_chat(self.gpt_model, messages, self.gpt_max_token, seed=trial,
                                                         n_response=5)

            for i, (response, context) in enumerate(zip(responses, contexts)):
                self.logging(context, logging.INFO)
                self.logging(response, logging.DEBUG)

                response_file_path = path.join(base_directory, f"{base_name}_branch_{i}.response.pkl")
                with open(response_file_path, 'wb') as f:
                    pickle.dump(response, f)

                context_file_path = path.join(base_directory, f"{base_name}_branch_{i}.context.pkl")
                with open(context_file_path, 'wb') as f:
                    pickle.dump(context, f)

                log_dict = {
                    'request': messages,
                    'response': response,
                }

                parsed_reward_function = parse_reward_function(response)

                # Save reward function to .py
                reward_file_path = path.join(base_directory, f"{base_name}_branch_{i}.py")
                with open(reward_file_path, 'w') as f:
                    f.write(parsed_reward_function)

                # Save the log to .json file
                log_file_path = path.join(base_directory, f"{base_name}_branch_{i}.json")
                with open(log_file_path, 'w') as f:
                    json.dump(log_dict, f, indent=4)

                self.save_data(response=response,
                               context=context,
                               base_directory=base_directory,
                               base_name=base_name,
                               messages=messages, branch=i)

            log_dict = {
                'selected_branch': f'branch_{index}'
            }
            log_file_path = path.join(self.reward_function_path, "selected_branch.json")
            with open(log_file_path, 'w') as f:
                json.dump(log_dict, f, indent=4)

            response = responses[index]
            context = contexts[index]
        else:
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

    def get_pe_prompt(self, pe: str):
        # files in the PE directory

        if pe == 'io':
            pe_file = path.join(self.file_path, PE_DIR, 'io.txt')
        elif pe == 'cot':
            pe_file = path.join(self.file_path, PE_DIR, 'cot.txt')
        elif pe == 'cotsc':
            pe_file = path.join(self.file_path, PE_DIR, 'cot.txt')
        elif pe == 'tot':
            pe_file = path.join(self.file_path, PE_DIR, 'cot.txt')
        elif pe == 'got':
            pe_file = path.join(self.file_path, PE_DIR, 'got.txt')
        else:
            warnings.warn(f"Unknown PE type: {pe}. Using the default PE file.")
            pe_file = path.join(self.file_path, PE_DIR, 'io.txt')

        pe_template = file_to_string(pe_file)

        if pe == 'tot':
            current_iteration = int((self.iteration_num - 1) // self.branch_factor + 1)
            total_iterations = int((self.config['total_iterations'] - 1) // self.branch_factor + 1)
            iteration_perc_str = "{:.1f}%".format(current_iteration / total_iterations * 100)
            pe_template = pe_template.format(
                curr_iteration_num=current_iteration,
                max_iteration_num=total_iterations,
                perc_iteration=iteration_perc_str
            )
        elif pe == 'got':
            cot_prompt = self.get_pe_prompt(pe='cot')

            if self.auxiliary_prompt_path is None:
                auxiliary_prompt = "- No auxiliary prompt is provided."
            else:
                auxiliary_prompt = read_file(self.auxiliary_prompt_path)

            pe_template = pe_template.format(
                cot_prompt=cot_prompt,
                eval_result=self.prev_eval_result,
                auxiliary_prompt=auxiliary_prompt,
            )

        else:
            iteration_perc_str = "{:.1f}%".format(self.config['iteration_num'] / self.config['total_iterations'] * 100)
            pe_template = pe_template.format(
                curr_iteration_num=self.config['iteration_num'],
                max_iteration_num=self.config['total_iterations'],
                perc_iteration=iteration_perc_str
            )
        pe_str = f"\n\n## Thought Tips\n{pe_template}\n"

        return pe_str

    def get_feature_prompt(self, feature: str):
        if feature == 'array':
            feature_file = read_file(path.join(self.file_path, FEATURE_DIR, 'array.txt'))
        elif feature == 'stats':
            feature_file = read_file(path.join(self.file_path, FEATURE_DIR, 'stats.txt'))
        else:
            raise ValueError(f"Unknown feature type: {feature}")
        return feature_file

    def get_feedback_prompt(self):
        feedback_str = file_to_string(self.config['feedback_path'])

        feedback_prompt = f"\n\n## Feedback\n{feedback_str}\n"

        return feedback_prompt

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
            result = run_validate(config)
        except Exception as e:
            code = read_file(reward_function_path)

            # get traceback from e
            tb_list = traceback.format_exception(type(e), e, e.__traceback__)

            string_lines = [line.strip() for line in tb_list if "File \"<string>\"" in line]
            filtered_traceback = ''.join(string_lines).strip()

            raise RewardExecutionException(code, f'{filtered_traceback}\n{e}')

        return result


def generate_reward(config: Config, generate_args: dict, return_error: bool = False):
    reward_generator = RewardGenerator(generate_args)
    reward_generator.set_execution_config(config)
    return reward_generator.run(return_error=return_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--shared_storage_path', type=str, default=abspath('.'))
    parser.add_argument('--postfix', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    parser.add_argument('--reward_functions_dir', type=str, default='RewardFunctions')
    parser.add_argument('--gpt_model', type=str, default='gpt-4o')
    parser.add_argument('--gpt_max_token', type=int, default=4096)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--current_inner', type=int, default=1)
    parser.add_argument('--task', type=str, default='alphabet')
    parser.add_argument('--available_tiles', type=list, default=list())
    parser.add_argument('--arbitrary_dataset', type=str, default='./example/random_dataset.txt')
    parser.add_argument('--n_codegen_trials', type=int, default=1)
    parser.add_argument('--n_codefix_trials', type=int, default=5)
    parser.add_argument('--iteration_num', type=int, default=1)
    parser.add_argument('--previous_reward_function', type=str, default=None)
    parser.add_argument('--map_width', type=int, default=16)
    parser.add_argument('--map_height', type=int, default=16)
    parser.add_argument('--feature', type=str, default='array')
    parser.add_argument('--feedback_path', type=str, default=None)
    parser.add_argument('--initial_reward_function', type=str, default=None)
    parser.add_argument('--target_character', type=str, default='A')
    parser.add_argument('--total_iterations', type=int, default=1)
    parser.add_argument('--pe', type=str, default='io')

    args = parser.parse_args()

    args = vars(args)

    train_config = TrainLLMConfig()
    init_config(train_config)

    reward_generator = RewardGenerator(args)
    reward_generator.set_execution_config(train_config)

    reward_generator.run()