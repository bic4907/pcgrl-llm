import logging
import os
import pprint
import textwrap

from tqdm import tqdm
import cv2
import numpy as np
from os.path import basename, join, dirname
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from debug.render_level import render_level
from debug.scenario_levels import AllLevels
from pcgrllm.evaluation.solution import SolutionEvaluator
from pcgrllm.task import TaskType
from pcgrllm.utils.storage import Iteration


def add_text_below_image(image, text, font_size=20, target_width=500):
    """Add wrapped text below an image with a black background."""
    # Path to the font file
    font_path = "Pretendard-Regular.ttf"  # Update this to the correct path
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        raise FileNotFoundError(f"Font file not found at {font_path}. Please provide a valid font file path.")

    draw = ImageDraw.Draw(image)

    # Wrap text to fit within the target width
    wrapped_text = textwrap.fill(text, width=int(target_width / (font_size / 2)))
    text_lines = wrapped_text.split("\n")

    # Calculate total text height
    text_height = sum(
        draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
        for line in text_lines
    )
    padding = 20  # Padding around the text
    total_height = image.height + text_height + padding

    # Create a new image
    new_image = Image.new("RGB", (image.width, total_height), (0, 0, 0))  # Black background
    new_image.paste(image, (0, 0))

    # Draw text on the new image
    draw = ImageDraw.Draw(new_image)
    y_text = image.height + 10
    for line in text_lines:
        # Calculate text width and height using textbbox
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]
        text_x = (image.width - line_width) // 2
        draw.text((text_x, y_text), line, fill=(255, 255, 255), font=font)  # Ensure white text is used
        y_text += line_bbox[3] - line_bbox[1]

    return new_image


def create_image_grid(images, grid_size, output_path=None, cell_size=(500, 500)):
    """Create a grid of images resized to the given cell size."""
    grid_width, grid_height = grid_size
    cell_width, cell_height = cell_size

    # Create a blank grid
    grid_img = Image.new("RGB", (cell_width * grid_width, cell_height * grid_height), (0, 0, 0))

    for idx, img in enumerate(images):
        # Resize image to fit cell size
        img_resized = img.resize(cell_size)
        x = (idx % grid_width) * cell_width
        y = (idx // grid_width) * cell_height
        grid_img.paste(img_resized, (x, y))

    if output_path:
        grid_img.save(output_path)
    return grid_img


if __name__ == '__main__':
    # Initialize logger
    logger = logging.getLogger(basename(__file__))
    logger.setLevel(logging.DEBUG)

    evaluator = SolutionEvaluator(logger=logger, task=TaskType.Scenario)
    base_path = join(dirname(__file__), '.cache')
    os.makedirs(base_path, exist_ok=True)
    example_path = join(base_path, 'iteration_1')
    os.makedirs(example_path, exist_ok=True)

    iteration = Iteration.from_path(path=example_path)
    os.makedirs(iteration.get_inference_dir(), exist_ok=True)
    os.makedirs(iteration.get_numpy_dir(), exist_ok=True)

    processed_images = []

    for idx, level in tqdm(enumerate(AllLevels[:])):
        # Render level as numpy array
        level_img = render_level(level, return_numpy=True, tile_size=32)
        # level_img = cv2.cvtColor(level_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        level_image_pil = Image.fromarray(level_img)  # Convert numpy to PIL Image

        # Save the level data
        np.save(join(iteration.get_numpy_dir(), f"level_{idx}_0.npy"), level)

        # Evaluate level
        result = evaluator.run(iteration=iteration, scenario_num="1", visualize=True, step_filter=idx)

        # Format result using pformat
        result_str = pprint.pformat(vars(result), indent=2, width=80)

        # Add text below the image
        level_image_with_text = add_text_below_image(level_image_pil, result_str)

        # Append processed image to the list
        processed_images.append(level_image_with_text)

    # Create a grid of images
    grid_output_path = join(base_path, "level_results_grid.png")
    grid_img = create_image_grid(processed_images, grid_size=(4, 4), output_path=grid_output_path)

    # Show the grid
    plt.figure(figsize=(20, 20))
    plt.imshow(grid_img)
    plt.axis('off')
    plt.show()

    print(f"Grid saved at {grid_output_path}")
