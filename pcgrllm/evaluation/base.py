import logging
from logging import Logger


class EvaluationResult:
    # Alphabet generation task
    similarity: float = 0
    diversity: float = 0
    sample_size: int = 0
    similarity_weight: float = 0.5
    diversity_weight: float = 0.5

    # Scenario generation task
    playability: float = 0      # If the player can reach to the door
    path_length: float = 0      # The path length from the player to the door
    solvability: float = 0      # if the player can reach the door with the key
    n_solutions: float = 0      # Number of solutions the player can reach the door
    loss_solutions: float = 0   # Loss of between the target and the current solution count
    acc_imp_tiles: float = 0        # (reachable of important tiles <-> prompt)
    exist_imp_tiles: float = 0      # (existence of important tiles <-> prompt)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key != 'total':  # Skip 'total' to avoid AttributeError
                setattr(self, key, value)

    @property
    def total(self):
        return (self.similarity_weight * self.similarity +
                self.diversity_weight * self.diversity)

    def __str__(self):
        return f"EvaluationResult(similarity={self.similarity}, diversity={self.diversity}, sample_size={self.sample_size})"

    def to_dict(self):
        return {
            'similarity': self.similarity,
            'diversity': self.diversity,
            'sample_size': self.sample_size
        }

    @staticmethod
    def from_dict(data: dict):
        return EvaluationResult(**data)

    def to_prompt(self):
        return f"Similarity: {self.similarity}, Diversity: {self.diversity}"

class LevelEvaluator:
    def __init__(self, logger: Logger = None):
        self.logger = logger

    def run(self):
        raise NotImplementedError

    def logging(self, message: str, level: int = logging.DEBUG):
        if self.logger:
            self.logger.log(level, message)