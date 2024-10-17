from os.path import abspath, join, basename, isdir
from glob import glob
from typing import List, Optional
import numpy as np
from PIL import Image

# Constants
ITERATION_PREFIX: str = 'iteration_'
INFERENCE_DIR: str = 'inference'
IMAGE_DIR: str = 'images'
NUMPY_DIR: str = 'numpy'


class ImageResource:
    """Represents an image resource."""
    def __init__(self, path: str) -> None:
        self.path: str = path

    def __str__(self) -> str:
        return f'ImageResource(path={self.path})'

    def load(self) -> Image.Image:
        """Loads the image as a PIL Image object."""
        return Image.open(self.path)


class NumpyResource:
    """Represents a numpy file resource."""
    def __init__(self, path: str) -> None:
        self.path: str = path

    def __str__(self) -> str:
        return f'NumpyResource(path={self.path})'

    def load(self, dtype = np.uint16) -> np.ndarray:
        """Loads the numpy file as a NumPy array."""
        return np.load(self.path, allow_pickle=True).astype(dtype)


class Iteration:
    """Represents an iteration containing images and numpy files inside the inference folder."""
    def __init__(self, iteration_num: int, root_path: str) -> None:
        self.iteration_num: int = iteration_num
        self.root_path: str = root_path

    def __str__(self) -> str:
        return (
            f"Iteration {self.iteration_num}\n"
            f"\tImages: {len(self.get_images())} files\n"
            f"\tNumpy Files: {len(self.get_numpy_files())} files"
        )

    @staticmethod
    def from_path(path: str) -> Optional['Iteration']:
        """Creates an Iteration object from the given path."""
        iteration_num = int(basename(path).split('_')[-1])
        return Iteration(iteration_num, path)

    def get_path(self) -> str:
        return self.root_path

    def get_inference_dir(self) -> str:
        return join(self.root_path, INFERENCE_DIR)

    def get_image_dir(self) -> str:
        return join(self.get_inference_dir(), IMAGE_DIR)

    def get_numpy_dir(self) -> str:
        return join(self.get_inference_dir(), NUMPY_DIR)

    def get_images(self) -> List[ImageResource]:
        image_paths = glob(join(self.get_image_dir(), '*.png'))
        return [ImageResource(path) for path in image_paths]

    def get_numpy_files(self) -> List[NumpyResource]:
        numpy_paths = glob(join(self.get_numpy_dir(), '*.npy'))
        return [NumpyResource(path) for path in numpy_paths]


class Storage:
    """Manages multiple iterations and their resources."""
    def __init__(self, path: str) -> None:
        self.root_path: str = abspath(path)

    def __str__(self) -> str:
        iteration_details = "\n".join(str(iteration) for iteration in self.get_iterations())
        return f"(Storage: {self.root_path})\n{iteration_details}"

    def get_iterations(self) -> List[Iteration]:
        iteration_nums = self._get_iteration_nums()
        return [
            Iteration(num, join(self.root_path, f"{ITERATION_PREFIX}{num}"))
            for num in iteration_nums
        ]

    def _get_iteration_nums(self) -> List[int]:
        iteration_regex = join(self.root_path, f'{ITERATION_PREFIX}*')
        iter_dirs = glob(iteration_regex)
        iter_dirs = [basename(d) for d in iter_dirs if isdir(d)]
        return [int(d.split('_')[-1]) for d in iter_dirs]

    def get_iteration(self, iteration_num: int) -> Optional[Iteration]:
        iteration_path = join(self.root_path, f"{ITERATION_PREFIX}{iteration_num}")
        if isdir(iteration_path):
            return Iteration(iteration_num, iteration_path)
        return None


def get_first_file(resource_list: List[ImageResource]) -> Optional[str]:
    """Returns the path of the first file from a resource list."""
    if resource_list:
        return resource_list[0].path
    return None


def main():
    # Set the test path
    test_path = "/Users/inchang/Desktop/pcgrl-llm/pcgrllm/example/binary_narrow-w-16_gpt_model-gpt-4o-gil_3"

    # Create Storage instance
    storage = Storage(test_path)

    # Iterate over each Iteration and get the first image and numpy file
    iterations = storage.get_iterations()
    for iteration in iterations:
        print(f"\nIteration {iteration.iteration_num}")

        # Get the first image file
        first_image_path = get_first_file(iteration.get_images())
        if first_image_path:
            print(f"First Image: {first_image_path}")
        else:
            print("No images found.")

        # Get the first numpy file
        first_numpy_path = get_first_file(iteration.get_numpy_files())
        if first_numpy_path:
            print(f"First Numpy File: {first_numpy_path}")
        else:
            print("No numpy files found.")

    # Print Storage summary
    print("\nStorage Summary:")
    print(storage)


if __name__ == "__main__":
    main()
