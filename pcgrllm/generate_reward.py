import copy
import datetime
import os, re, ast, sys, time, argparse, json, pickle
import pprint
import shutil
import multiprocessing
import warnings
from copy import deepcopy
import random

import pandas as pd
import numpy as np
from os import path
import tempfile
from os.path import abspath, basename
import logging
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from conf.config import TrainLLMConfig, TrainConfig, Config
from pcgrllm.utils.exceptions import RewardExecutionException, RewardParsingException
from pcgrllm.validate_reward import run_validate, read_file

logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


from pcgrllm.llm_client.llm import ChatContext, UnifiedLLMClient
from pcgrllm.llm_client.utils import *

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))

PE_DIR = 'pe'
FEATURE_DIR = 'feature'

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
        self.n_inner = config.get('n_inner', 3)

        self.current_inner = config.get('current_inner', 1)

        self.n_outer = config.get('n_outer', 3)
        self._current_trial = 0
        self.trial_count = config.get('trial_count', 3)
        self.iteration_num = config.get('iteration_num', 1)
        self.branch_factor = config.get('branch_factor', None)
        if self.branch_factor is None:
            self.total_iterations = config.get('total_iterations', 3)
        else:
            self.total_iterations = config.get('total_iterations', 3) * self.branch_factor
        self.reference_csv = config.get('reference_csv', 'random_dataset.txt')
        self.arbitrary_dataset = config.get('arbitrary_dataset', 'arbitrary_dataset.txt')
        self.file_path = path.join(self.shared_storage_path, 'prompt')
        self.example_path = path.join(self.shared_storage_path, 'example')
        if self.reference_csv == 'random_dataset.txt':
            self.reference_csv = path.join(self.example_path, self.reference_csv)
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
            'inner_loop': self.current_inner,
            'n_inner': self.n_inner,
            'trial': self._current_trial if hasattr(self, '_current_trial') else -1,  # Assuming self._current_trial is a class attribute
            'trial_count': self.trial_count if hasattr(self, 'trial_count') else -1,
        }

        # Define the prefix format
        prefix = '[iter: {outer_loop}, self-alignment: {inner_loop} / {n_inner}, trial: {trial} / {trial_count}]'.format(**info_dict)

        # Split the message by line breaks and log each line with the prefix
        message = str(message)
        for line in message.splitlines():
            formatted_message = f'{prefix} {line}'
            logger.log(level, formatted_message)

    def _prepare_initial_reward_function(self):
        # Copy the initial reward function to the reward function path
        reward_file_name = basename(self.initial_reward_function)

        initial_reward_function_path = path.join(self.reward_function_path, reward_file_name)
        # Copy
        shutil.copy(self.initial_reward_function, initial_reward_function_path)

        self.generating_function_path = initial_reward_function_path
        self.previous_reward_function = file_to_string(self.generating_function_path)
        self.logging(f"Copied the initial reward function to the reward function path: {self.initial_reward_function} -> {initial_reward_function_path}", logging.INFO)





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
                responses = client.call_model(ctx, messages, model=model, seed=seed, n_response=n_response, temperature=temperature)
                if n_response != 1:
                    generated_responses = [response[0] for response in responses]
                    generated_contexts = [response[1] for response in responses]
                    index = self.response_cluster(generated_responses, n_clusters=3)
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

    def run(self):

        while self.current_inner <= self.n_inner:
            reward_function_name = f"{self.postfix}_inner_{self.current_inner}"
            is_success = False

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

            for i_trial in range(1, self.trial_count + 1):

                self._current_trial = i_trial
                basename = f"{reward_function_name}_trial_{i_trial}"

                self.logging(f"Generating reward function: {basename}", logging.INFO)

                if self.current_inner == 1 and self.iteration_num == 1:
                    self.logging(f'Calling the zero-shot generation function. (len(error): {len(generating_function_error) if generating_function_error else 0})')
                    generating_function_path = self.first_user_response(basename=basename,
                                                                        generating_function_path=generating_function_path,
                                                                        generating_function_error=generating_function_error,
                                                                        trial=i_trial)


                    self.logging(f'Called the first_user_response function')
                else:
                    self.logging(f'Calling the inner-loop generation function. (len(error): {len(generating_function_error) if generating_function_error else 0})')
                    generating_function_path = self.first_user_response(basename=basename,
                                                                        generating_function_path=generating_function_path,
                                                                        generating_function_error=generating_function_error,
                                                                        trial=i_trial)
                    # TODO implement second_user_function with the feedback prompts
                    self.logging(f'Called the second_user_response function')


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
                    is_success = False


                if is_success:
                    self.previous_reward_function = file_to_string(generating_function_path)
                    self.previous_reward_function_path = generating_function_path

                    break
                else:
                    generating_function_error = error_message

            if not is_success:
                self.logging(f"Failed to generate the reward function: {reward_function_name}", logging.WARNING)
                return False

            self.current_inner += 1

        # Save the reward function to the file

        reward_function_string = file_to_string(generating_function_path)
        reward_function_file_path = path.join(self.reward_function_path, f"{reward_function_name}.py")
        with open(reward_function_file_path, 'w') as f:
            self.logging(f"Saving the reward function to the file: {reward_function_file_path}", logging.INFO)
            f.write(reward_function_string)

        return reward_function_file_path

    def response_cluster(self, responses, n_clusters=2):
        # set environment variable for

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = model.encode(responses)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

        largest_cluster = max(clusters.values(), key=len)
        largest_cluster_embeddings = embeddings[largest_cluster]

        cluster_center = np.mean(largest_cluster_embeddings, axis=0)
        index = min(largest_cluster, key=lambda i: np.linalg.norm(embeddings[i] - cluster_center))
        return index

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

        level_shape_str = f"({self.config['map_height']}, {self.config['map_width']})"

        prompt = prompt.format(
            array_shape=level_shape_str,
            stats_keys='DIAMETER = 0, N_REGIONS = 1',
            tile_enum='EMPTY = 1, WALL = 2'
        )

        return prompt

    def _get_stats_feature(self):
        prompt = self.get_feature_prompt('stats')
        return prompt

    def save_data(self, response: str, context: list, messages: list, basename: str = 'reward', branch: int=None):

        self.logging(context, logging.INFO)
        self.logging(response, logging.DEBUG)
        if branch is None:
            file_name = f"{basename}"
        else:
            file_name = f"{basename}_branch_{branch}"

        response_file_path = path.join(self.reward_function_path, f"{file_name}.response.pkl")
        with open(response_file_path, 'wb') as f:
            pickle.dump(response, f)

        context_file_path = path.join(self.reward_function_path, f"{file_name}.context.pkl")
        with open(context_file_path, 'wb') as f:
            pickle.dump(context, f)

        parsed_reward_function = parse_reward_function(response)

        log_dict = {
            'request': messages,
            'response': response,
        }

        # Save reward function to .py
        reward_file_path = path.join(self.reward_function_path, f"{file_name}.py")
        with open(reward_file_path, 'w') as f:
            f.write(parsed_reward_function)

        # Save the log to .json file
        log_file_path = path.join(self.reward_function_path, f"{file_name}.json")
        with open(log_file_path, 'w') as f:
            json.dump(log_dict, f, indent=4)

        return reward_file_path


    def first_user_response(self, basename: str = 'reward', generating_function_path: str = None, generating_function_error: str = None, trial=1):

        self.initial_system = self.initial_system.format(
            i='{i}',
            reward_signature=self.reward_template,
        )

        # Add jax code tips prompt
        self.initial_system += self.jax_code_tips_prompt
        self.initial_system += '\n'
        self.initial_system += self.reward_code_tips_prompt

        initial_user = copy.deepcopy(self.initial_user)


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


              _character=self._execution_config.target_character,
                few_shot_code_string=sample_code,
                reward_function_inputs=reward_function_inputs,
                target_character=self.config['target_character'],
                thought_tips=self.get_pe_prompt(self.pe),
            )


        elif self.config['feedback_path'] is not None: # Feedback available

            reward_code = file_to_string(generating_function_path)

            sample_code = """
               ## Previous Reward Code
               Here is the previous reward function that you have generated. However, this code has an error. Please fix the error and generate the reward function again.
               ```python
               {reward_code_string}
               ```
               
               Feedback:
               {feedback}

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

            initial_user = initial_user.format(
                target_character=self.config['target_character'],
                few_shot_code_string=sample_code,
                reward_function_inputs=reward_function_inputs,
                thought_tips=self.get_pe_prompt(self.pe),
            )


        # 피드백 받는 부분 작성 필요함


        messages = [
            {"role": "system", "content": self.initial_system},
            {"role": "user", "content": initial_user}
        ]

        self.logging(f'Input to the reward generation model:\n{json.dumps(messages, indent=2)}', logging.DEBUG)

        if self.pe == 'cotsc':
            responses, contexts, index = self.start_chat(self.gpt_model, messages, self.gpt_max_token, seed=trial, n_response=5)

            for i, (response, context) in enumerate(zip(responses, contexts)):
                self.logging(context, logging.INFO)
                self.logging(response, logging.DEBUG)

                reward_file_path = self.save_data(response=response, context=context, messages=messages, branch=i)

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

        reward_file_path = self.save_data(response=response, context=context, messages=messages)

        return reward_file_path

    def get_pe_prompt(self, pe: str):
        # files in the PE directory

        if self.pe == 'io':
            pe_file = path.join(self.file_path, PE_DIR, 'io.txt')
        elif self.pe == 'cot':
            pe_file = path.join(self.file_path, PE_DIR, 'cot.txt')
        elif self.pe == 'cotsc':
            pe_file = path.join(self.file_path, PE_DIR, 'cot.txt')
        elif self.pe == 'tot':
            pe_file = path.join(self.file_path, PE_DIR, 'cot.txt')
        elif self.pe == 'got':
            pe_file = path.join(self.file_path, PE_DIR, 'cot.txt')
        else:
            warnings.warn(f"Unknown PE type: {pe}. Using the default PE file.")
            pe_file = path.join(self.file_path, PE_DIR, 'io.txt')

        pe_template = file_to_string(pe_file)
        if self.pe == 'tot':
            current_iteration = int((self.iteration_num - 1) // self.branch_factor + 1)
            total_iterations = int((self.config['total_iterations'] - 1) // self.branch_factor + 1)
            iteration_perc_str = "{:.1f}%".format(current_iteration / total_iterations * 100)
            pe_template = pe_template.format(
                curr_iteration_num=current_iteration,
                max_iteration_num=total_iterations,
                perc_iteration=iteration_perc_str
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

    # FIX ME: Implement the function
    # def second_user_response(self, basename: str = 'reward', generating_function_path: str = None, generating_function_error: str = None, trial=1):
    #     playtesting_result = ""
    #     parsed_code = ast.parse(self.previous_reward_function)
    #     error_message = None
    #
    #     sampled_data_path = abspath(path.join(self.reward_function_path, f"{basename}.data.json"))
    #     preprocess_dataset(self.arbitrary_dataset, sampled_data_path)
    #     # read the sampled data and save the lines with array
    #
    #     # This section is for module test of the reward function
    #
    #     sample_data_arr = list()
    #     with open(sampled_data_path, 'r') as file:
    #         for line in file:
    #             sample_data_arr.append(preprocessing(json.loads(line)))
    #
    #     try:
    #         for node in ast.walk(parsed_code):
    #             if isinstance(node, ast.FunctionDef) and node.name == 'compute_reward':
    #
    #                 inner_node_list = list()
    #
    #                 for inner_node in ast.iter_child_nodes(node):
    #                     if isinstance(inner_node, ast.FunctionDef):
    #                         inner_node_list.append(inner_node.name)
    #
    #                         nested_function_code = astor.to_source(inner_node)
    #                         exec(nested_function_code, globals())
    #
    #                 self.logging(f'Founded ({len(inner_node_list)}) sub reward functions: {inner_node_list} and sample data: {len(sample_data_arr)}', logging.DEBUG)
    #
    #                 result_dict = dict()
    #
    #                 for inner_node in ast.iter_child_nodes(node):
    #                     if isinstance(inner_node, ast.Assign):
    #                         if isinstance(inner_node.value, ast.Constant):
    #                             exec(astor.to_source(inner_node), globals())
    #
    #                     if isinstance(inner_node, ast.FunctionDef):
    #
    #                         try:
    #                             result_list = list()
    #
    #                             # Input the arbitrary game data to the sub reward function
    #                             for kwarg in sample_data_arr:
    #                                 result = globals()[inner_node.name](kwarg['Current'])
    #                                 result_list.append(result)
    #
    #                             average_value = sum(result_list) / len(result_list)
    #                             standard_deviation = sum([(x - average_value) ** 2 for x in result_list]) / len(
    #                                 result_list) ** 0.5
    #
    #                             result_dict[inner_node.name] = {'Average': average_value, 'Standard deviation': standard_deviation}
    #
    #                         except:
    #                             error_msg = traceback.format_exc()
    #                             result_dict[inner_node.name] = {'Error': error_msg }
    #                             self.logging(error_msg, logging.ERROR)
    #
    #                 # End of loop
    #
    #         # Result for sub reward functions
    #         if len(result_dict) > 0:
    #             _playtesting_result = "[Sub-reward Output Analysis]\n"
    #
    #             for node_name, item in result_dict.items():
    #                 if isinstance(item, dict):
    #                     _playtesting_result += f"({node_name}) "  # node_name 한 번만 출력
    #                     result_line = []
    #                     for key, value in item.items():
    #                         # 실수는 소수점 3자리까지 출력
    #                         if isinstance(value, float):
    #                             value_str = f"{value:.3f}"
    #                         else:
    #                             value_str = str(value)
    #
    #                         # key-value 쌍을 리스트에 추가
    #                         result_line.append(f"{key}: {value_str}")
    #
    #                     # 각 node_name에 대한 결과를 한 줄로 출력
    #                     _playtesting_result += '  '.join(result_line) + '\n'
    #
    #
    #             playtesting_result =_playtesting_result.strip()  # 마지막 공백 제거
    #
    #         # Result for the main reward function
    #         required_node_list = ['agent_0', 'agent_1', 'agent_2', 'agent_3']
    #         # 부족한 항목과 추가된 항목을 비교
    #         missing_nodes = set(required_node_list) - set(inner_node_list)
    #         extra_nodes = set(inner_node_list) - set(required_node_list)
    #
    #         # 메시지 작성
    #         node_message = ""
    #
    #         if extra_nodes:
    #             node_message += f"\nExtra nodes found, please remove: {', '.join(extra_nodes)}\n"
    #         if missing_nodes:
    #             node_message += f"\nMissing nodes: {', '.join(missing_nodes)}\n"
    #
    #         playtesting_result += node_message
    #
    #     except:
    #         self.logging(traceback.format_exc(), logging.ERROR)
    #         error_message = traceback.format_exc()
    #
    #     # End of the module t est of the reward function
    #
    #     # Start of the execution test of the reward function
    #     reward_mean, reward_std, success_rate = self.execute_reward_functions_parallel(self.previous_reward_function_path, state_dicts=sample_data_arr)
    #     playtesting_result += '\n[Total Reward Analysis]\n'
    #     playtesting_result += 'Average: {:.3f}\n'.format(reward_mean)
    #     playtesting_result += 'Standard deviation: {:.3f}\n'.format(reward_std)
    #     playtesting_result += 'Success Rate: {:.1f}%\n'.format(success_rate)
    #
    #     self.logging(playtesting_result, logging.DEBUG)
    #
    #     if generating_function_error is not None:
    #         error_description = " The previous reward function has an error. Below is the error message. Please generate the reward function again with attention to error.\n" + generating_function_error
    #     elif error_message is not None:
    #         error_description = " The previous reward function has an error. Below is the error message. Please generate the reward function again with attention to error.\n" + error_message
    #     else:
    #         error_description = ""
    #
    #     self.second_user = file_to_string(path.join(self.file_path, "second_user.txt"))
    #     self.second_user = self.second_user.format(
    #         i='{i}',
    #         previous_reward_function=self.previous_reward_function,
    #         error_description=error_description,
    #         playtesting_result=playtesting_result
    #     )
    #     messages = [
    #         {"role": "system", "content": self.initial_system},
    #         {"role": "user", "content": self.second_user}
    #     ]
    #
    #     response, context = self.start_chat(self.gpt_model, messages, self.gpt_max_token, seed=trial)
    #     self.logging(context, logging.INFO)
    #     self.logging(response, logging.DEBUG)
    #
    #     os.makedirs(self.reward_function_path, exist_ok=True)
    #     response_file_path = path.join(self.reward_function_path, f"{basename}.response.pkl")
    #     with open(response_file_path, 'wb') as f:
    #         pickle.dump(response, f)
    #
    #     context_file_path = path.join(self.reward_function_path, f"{basename}.context.pkl")
    #     with open(context_file_path, 'wb') as f:
    #         pickle.dump(context, f)
    #
    #     log_dict = {
    #         'request': messages,
    #         'response': response,
    #     }
    #
    #     # Save reward function to .py
    #     reward_file_path = path.join(self.reward_function_path, f"{basename}.py")
    #     with open(reward_file_path, 'w') as f:
    #         f.write(response)
    #
    #     # Save the log to .json file
    #     log_file_path = path.join(self.reward_function_path, f"{basename}.json")
    #     with open(log_file_path, 'w') as f:
    #         json.dump(log_dict, f, indent=4)
    #
    #     return reward_file_path

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
            raise RewardExecutionException(code, e)

        return result

def generate_reward(config: Config, generate_args: dict):
    reward_generator = RewardGenerator(generate_args)
    reward_generator.set_execution_config(config)
    return reward_generator.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--shared_storage_path', type=str, default='.')
    parser.add_argument('--postfix', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    parser.add_argument('--reward_functions_dir', type=str, default='RewardFunctions')
    parser.add_argument('--gpt_model', type=str, default='llama3-70b-instruct')
    parser.add_argument('--gpt_max_token', type=int, default=4096)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--current_inner', type=int, default=1)
    parser.add_argument('--n_inner', type=int, default=1)
    parser.add_argument('--n_outer', type=int, default=1)
    parser.add_argument('--reference_csv', type=str, default='random_dataset.txt')
    parser.add_argument('--arbitrary_dataset', type=str, default='./example/random_dataset.txt')
    parser.add_argument('--trial_count', type=int, default=10)
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
    reward_generator = RewardGenerator(args)
    reward_generator.run()