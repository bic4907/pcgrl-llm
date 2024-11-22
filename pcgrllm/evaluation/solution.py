import numpy as np
from os.path import dirname, join, basename
from typing import Tuple
from pcgrllm.evaluation.base import *
from pcgrllm.scenario_preset import ScenarioPreset
from pcgrllm.utils.storage import Iteration



class SolutionEvaluator(LevelEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval_level(self, level: np.ndarray, scenario_num: str) -> Tuple[float, float]:
        '''
        # Scenario generation task
        playability: float = 0  # If the player can reach to the door
        path_length: float = 0  # The path length from the player to the door
        solvability: float = 0  # if the player can reach the door with the key
        n_solutions: float = 0  # Number of solutions the player can reach the door
        loss_solutions: float = 0  # Loss of between the target and the current solution count
        acc_imp_tiles: float = 0  # (reachable of important tiles <-> prompt)
        exist_imp_tiles: float = 0  # (existence of important tiles <-> prompt)
        '''


        scenario = ScenarioPreset().scenarios[scenario_num]


        # TODO Check whether there is important tile in the level.


        return EvaluationResult(
            task=self.task,
            playability=0,
            path_length=0,
            solvability=0,
            n_solutions=0,
            loss_solutions=0,
            acc_imp_tiles=0,
            exist_imp_tiles=0,
            sample_size=1)

    def run(self, iteration: Iteration, scenario_num: str, visualize: bool = False) -> EvaluationResult:
        numpy_files = iteration.get_numpy_files()

        # run evaluation with each numpy file
        results = []
        for numpy_file in numpy_files:
            level = numpy_file.load()

            result = self.eval_level(level, scenario_num=scenario_num)
            results.append(result)

        # Calculate the average of the results
        playability = np.mean([result.playability for result in results])
        path_length = np.mean([result.path_length for result in results])
        solvability = np.mean([result.solvability for result in results])
        n_solutions = np.mean([result.n_solutions for result in results])
        loss_solutions = np.mean([result.loss_solutions for result in results])
        acc_imp_tiles = np.mean([result.acc_imp_tiles for result in results])
        exist_imp_tiles = np.mean([result.exist_imp_tiles for result in results])
        sample_size = len(results)

        return EvaluationResult(
            task=self.task,
            playability=playability,
            path_length=path_length,
            solvability=solvability,
            n_solutions=n_solutions,
            loss_solutions=loss_solutions,
            acc_imp_tiles=acc_imp_tiles,
            exist_imp_tiles=exist_imp_tiles,
            sample_size=sample_size)



if __name__ == '__main__':
    # Initialize logger
    logger = logging.getLogger(basename(__file__))
    logger.setLevel(logging.DEBUG)

    evaluator = SolutionEvaluator(logger=logger)


    base_path = join(dirname(__file__), 'example')
    # Define the path for the iteration folder
    example_path = join(base_path, 'A', 'iteration_1')

    # Load the iteration
    iteration = Iteration.from_path(path=example_path)

    # Run the evaluator with visualization enabled/disabled
    result = evaluator.run(iteration=iteration, scenario_num="1", visualize=True)

    print(result)