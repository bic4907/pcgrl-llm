from os.path import dirname, join, basename

from pcgrllm.evaluation.base import *
from pcgrllm.utils.storage import Iteration


class ViTEvaluator(LevelEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run(self, iteration: Iteration, target_character: str, visualize: bool = False) -> EvaluationResult:
        image_files = iteration.get_images()
        numpy_files = iteration.get_numpy_files()


        return EvaluationResult(similarity=0, sample_size=len(image_files))


def run_cross_validation(evaluator, letters, base_path, visualize=False):
    """
    Run cross-validation for a set of letters, where each letter is evaluated against others.
    The results are sorted by similarity score for each target letter.
    """
    results_dict = {target_letter: [] for target_letter in letters}  # Dictionary to store results for each target letter

    for target_letter in letters:
        for evaluate_letter in letters:

            # Define the path for the iteration folder
            example_path = join(base_path, evaluate_letter, 'iteration_1')

            # Load the iteration
            iteration = Iteration.from_path(path=example_path)

            # Run the evaluator with visualization enabled/disabled
            result = evaluator.run(iteration=iteration, target_character=target_letter, visualize=visualize)

            # Store the result for sorting later
            results_dict[target_letter].append((evaluate_letter, result.similarity))

    # Sort and print the results for each target letter
    for target_letter, evaluations in results_dict.items():
        # Sort evaluations by similarity score in descending order (higher is better)
        sorted_evaluations = sorted(evaluations, key=lambda x: x[1], reverse=True)

        print(f"\nSorted Results for target letter '{target_letter}':")
        for evaluate_letter, similarity in sorted_evaluations:
            print(f"Evaluated Letter: {evaluate_letter}, Similarity Score: {similarity:.4f}")


if __name__ == '__main__':
    # Initialize logger
    logger = logging.getLogger(basename(__file__))
    logger.setLevel(logging.DEBUG)

    evaluator = HeuristicEvaluator(logger=logger)

    # Define the letters and the base path for the evaluations
    letters = ['A', 'B', 'H']
    base_path = join(dirname(__file__), 'example')

    # Run the cross-validation and show sorted results with visualization enabled
    run_cross_validation(evaluator, letters, base_path, visualize=True)
