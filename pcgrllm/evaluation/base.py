import logging
import random
from logging import Logger


from pcgrllm.task import TaskType




class EvaluationResult:

    # Common
    task: TaskType
    sample_size: int = 0

    # Alphabet generation task
    similarity: float = 0
    diversity: float = 0
    attr_alphabet = ['similarity', 'diversity']

    # Scenario generation task
    playability: float = 0      # If the player can reach to the door
    path_length: float = 0      # The path length from the player to the door
    solvability: float = 0      # if the player can reach the door with the key
    n_solutions: float = 0      # Number of solutions the player can reach the door
    loss_solutions: float = 0   # Loss of between the target and the current solution count
    acc_imp_perc: float = 0        # (reachable of important tiles <-> prompt)
    exist_imp_perc: float = 0      # (existence of important tiles <-> prompt)
    attr_scenario = ['playability', 'path_length', 'solvability', 'n_solutions', 'loss_solutions', 'acc_imp_perc', 'exist_imp_perc']

    def __init__(self, task: TaskType, **kwargs):

        self.task = task

        for key, value in kwargs.items():
            if key != 'total':  # Skip 'total' to avoid AttributeError
                setattr(self, key, value)


    @property
    def total(self):
        if self.task == TaskType.Alphabet:
            return self.similarity
        elif self.task == TaskType.Scenario:
            return self.solvability


    def __str__(self):
        result_dict = self.to_dict()
        result_dict = ', '.join([f"{key}={value}" for key, value in result_dict.items()])
        return f"EvaluationResult({result_dict})"

    def to_dict(self):

        if self.task == TaskType.Alphabet:

            result_dict = {
                'similarity': float(self.similarity),
                'diversity': float(self.diversity),
            }
        elif self.task == TaskType.Scenario:
            result_dict = {
                'playability': float(self.playability),
                'path_length': float(self.path_length),
                'solvability': float(self.solvability),
                'n_solutions': float(self.n_solutions),
                'loss_solutions': float(self.loss_solutions),
                'acc_imp_perc': float(self.acc_imp_perc),
                'exist_imp_perc': float(self.exist_imp_perc),
            }
        else:
            raise ValueError(f"Invalid task type: {self.task}")

        # Add common attributes
        result_dict['task'] = self.task
        result_dict['sample_size'] = int(self.sample_size)

        # dict_obj = {k: float(v) for k, v in dict_obj.items()}
        return result_dict


    @staticmethod
    def from_dict(data: dict):
        return EvaluationResult(**data)

    def to_prompt(self):
        return f"Similarity: {self.similarity}, Diversity: {self.diversity}"

    def sample(self) -> 'EvaluationResult':
        result = EvaluationResult(self.task)

        for key, value in __dict__.items():
            if key not in ['sample_size', 'total', 'task']:
                setattr(result, key, random.random())

        return result


class LevelEvaluator:
    def __init__(self, task: TaskType, logger: Logger = None):
        self.task = task
        self.logger = logger

    def run(self):
        raise NotImplementedError

    def logging(self, message: str, level: int = logging.DEBUG):
        if self.logger:
            self.logger.log(level, message)