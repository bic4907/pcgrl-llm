import logging
from logging import Logger


class EvaluationResult:

    total: float

    similarity: float = -1
    diversity: float = -1

    sample_size: int = 0

    similarity_weight: float = 0.5
    diversity_weight: float = 0.5

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def total(self):
        return (self.similarity_weight * self.similarity + \
                self.diversity_weight * self.diversity)


    def __str__(self):
        return f"EvaluationResult(similarity={self.similarity}, diversity={self.diversity}, sample_size={self.sample_size})"

    def to_dict(self):
        return {
            'similarity': self.similarity,
            'diversity': self.diversity,
            'sample_size': self.sample_size,
            'total': self.total
        }

    @staticmethod
    def from_dict(data: dict):
        return EvaluationResult(**data)


class LevelEvaluator:
    def __init__(self, logger: Logger = None):
        self.logger = logger

    def run(self):
        raise NotImplementedError

    def logging(self, message: str, level: int = logging.DEBUG):
        if self.logger:
            self.logger.log(level, message)