import os
import sys
import json
import argparse
import logging
import warnings
from copy import deepcopy

import numpy as np
from PIL import Image
from math import ceil, sqrt
from io import BytesIO
import base64
from os.path import abspath, basename, join, dirname


from conf.config import Config
from envs.probs.binary import BinaryTiles
from pcgrllm.llm_client.llm import UnifiedLLMClient, ChatContext
from pcgrllm.utils.storage import Storage

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


class FeedbackInputType:
    Array = 'array'
    Image = 'image'

class FeedbackGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.storage = Storage(config['exp_path'])
        self.client = UnifiedLLMClient()

        self.logging(f'FeedbackGenerator initializing with config: {config}', level=logging.INFO)

        self.iteration = self.storage.get_iteration(config['iteration'])
        assert self.iteration is not None, f"Iteration {config['iteration']} not found in {self.config['exp_path']}."

        self._system_template = open(join(dirname(__file__), 'prompt', self.config.task, 'feedback_system.txt'), 'r').read()
        self._user_template = open(join(dirname(__file__), 'prompt', self.config.task, 'feedback_user.txt'), 'r').read()

        # Create feedback folder inside exp_path

        self.feedback_dir = join(self.iteration.get_path(), 'feedback')
        os.makedirs(self.feedback_dir, exist_ok=True)

        # Paths for logs and images
        self.log_path = join(self.feedback_dir, f'feedback_log_iter_{config["iteration"]}.json')
        self.grid_image_path = join(self.feedback_dir, 'image_input.png')

    def run(self):
        self.logging(f'FeedbackGenerator config: {self.config}', level=logging.INFO)

        if self.config['input_type'] == FeedbackInputType.Array:
            response = self.run_text_model()
        elif self.config['input_type'] == FeedbackInputType.Image:
            response = self.run_vision_model()
        else:
            self.logging('Invalid input type', level=logging.ERROR)
            raise ValueError('Invalid input type')

        # save to the iteration_dir and return the path
        feedback_path = join(self.feedback_dir, 'feedback.txt')
        with open(feedback_path, 'w') as f:
            f.write(response)

        return feedback_path

    def _get_system_message(self) -> str:
        prompt = deepcopy(self._system_template)
        return prompt

    def _get_user_message(self, input_type: FeedbackInputType, content: str = '') -> str:
        user_prompt = deepcopy(self._user_template)


        reward_function_prompt = ''
        if self.config['reward_function'] != None:
            # read the reward function and make a prompt
            with open(self.config['reward_function'], 'r') as f:
                reward_function = f.read()

            reward_function_prompt = f'Please provide feedback on the reward function:\n\n{reward_function}\n\n'

        if self.config['condition_prompt'] is None:
            warnings.warn('Condition prompt is not provided. Please provide a condition prompt.')
            self.config['condition_prompt'] = 'N/A'

        if input_type == FeedbackInputType.Array:
            available_tiles = set(self.config['available_tiles']) | {BinaryTiles.EMPTY, BinaryTiles.WALL}

            # Filter the enum members based on available_tiles
            tile_enum = ', '.join(f"{tile.name} = {tile.value}" for tile in BinaryTiles if tile in available_tiles)
            content = f'Available tiles: {tile_enum}\n\n{content}'

        user_prompt = user_prompt.format(
            evaluation_criteria=self.config['condition_prompt'] ,
            reward_function_prompt=reward_function_prompt,
            content=content
        )

        return user_prompt

    def run_text_model(self):
        numpy_files = self.iteration.get_numpy_files()

        numpy_str = ''
        for idx, numpy_file in enumerate(numpy_files):
            numpy_data = numpy_file.load()
            numpy_str += f'Level Id {idx + 1}:\n'
            numpy_str += f'{str(numpy_data)}\n\n'


        ctx = ChatContext()
        messages = [
            {"role": "system", "content": self._get_system_message()},
            {"role": "user", "content": self._get_user_message(input_type=self.config['input_type'],
                                                               content=numpy_str)},
        ]

        self.logging(f'Model input:\n{messages}')

        response, ctx = self.client.call_model(ctx, messages, model=self.config['gpt_model'])[0]

        self.save_log(ctx)

        return response

    def run_vision_model(self):
        images = [Image.open(image.path) for image in self.iteration.get_images()]
        grid_image = self.create_image_grid(images)

        # Save grid image to feedback folder
        grid_image.save(self.grid_image_path)
        self.logging(f'Saved grid image to: {self.grid_image_path}')

        # Convert the grid image to Base64
        base64_image = self.image_to_base64(grid_image)

        ctx = ChatContext()

        messages = [
            {"role": "system", "content": self._get_system_message()},
            {"role": "user", "content": [
                {"type": "text", "text": self._get_user_message(self.config['input_type'])},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }}
            ]}
        ]

        self.logging(f'Model input:\n{messages}')

        response, ctx = self.client.call_model(ctx, messages, model=self.config['gpt_model'])[0]
        self.save_log(ctx)

        return response

    def create_image_grid(self, images, padding=10, bg_color=(255, 255, 255)):
        """Create a grid image with the provided images."""
        num_images = len(images)
        grid_size = ceil(sqrt(num_images))

        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        max_height = max(heights)

        grid_width = grid_size * max_width + (grid_size - 1) * padding
        grid_height = grid_size * max_height + (grid_size - 1) * padding
        grid_image = Image.new("RGB", (grid_width, grid_height), bg_color)

        for idx, img in enumerate(images):
            row, col = divmod(idx, grid_size)
            x = col * (max_width + padding)
            y = row * (max_height + padding)
            grid_image.paste(img, (x, y))

        return grid_image

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to Base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def save_log(self, ctx: ChatContext):
        """Save the chat context log to the feedback folder."""
        with open(self.log_path, 'w') as f:
            f.write(json.dumps(ctx.to_json(), indent=4))
        self.logging(f'Feedback log saved to {self.log_path}')

    def logging(self, message, level=logging.DEBUG):
        """Log messages with iteration-specific prefix."""
        prefix = f"[iter: {self.config['iteration']}]"
        for line in str(message).splitlines():
            logger.log(level, f"{prefix} {line}")


def generate_feedback(config: Config, generate_args: dict):
    feedback_generator = FeedbackGenerator(generate_args)
    return feedback_generator.run()


def main():
    parser = argparse.ArgumentParser()
    example_exp_path = abspath('pcgrllm/example/binary_narrow-w-16_gpt_model-gpt-4o-gil_3')

    parser.add_argument('--condition_prompt', type=str, default='Is the level looks like an alphabet letter "A"?')
    parser.add_argument('--reward_function', type=str, default=None)
    parser.add_argument('--exp_path', type=str, default=example_exp_path)
    parser.add_argument('--input_type', type=str, default=FeedbackInputType.Array,
                        choices=[FeedbackInputType.Array, FeedbackInputType.Image])
    parser.add_argument('--iteration', type=int, default=1)
    args = vars(parser.parse_args())

    # print('Vision model feedback')
    args['input_type'] = FeedbackInputType.Image
    feedback_generator = FeedbackGenerator(args)
    output = feedback_generator.run()

if __name__ == "__main__":
    # Run the image and text model feedback processes
    main()
