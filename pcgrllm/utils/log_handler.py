import os
import wandb
import pandas as pd
from pcgrllm.utils.logger import get_wandb_name
from tensorboardX import SummaryWriter

# Base logging handler
class BaseLoggingHandler:
    def __init__(self, **kwargs):
        self.train_start_time = None
        self.steps_prev_complete = 0
        self.config = kwargs.get("config", None)
        self.logger = kwargs.get("logger", None)

    def set_start_time(self, start_time):
        """Sets the start time for training."""
        self.train_start_time = start_time

    def set_steps_prev_complete(self, steps):
        """Sets the previous steps completed."""
        self.steps_prev_complete = steps

    def log(self, metric, t):
        """Logs the metric."""
        raise NotImplementedError



# TensorBoard logging handler
class TensorBoardLoggingHandler(BaseLoggingHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.writer = SummaryWriter(self.config.exp_dir)

    def log(self, metric, t):
        for key, val in metric.items():
            self.writer.add_scalar(key, val, t)



# wandb logging handler
class WandbLoggingHandler(BaseLoggingHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.config.wandb_key and self.config.wandb_project:
            wandb.login(key=self.config.wandb_key)
            wandb.init(project=self.config.wandb_project, name=get_wandb_name(self.config), save_code=True)
            wandb.config.update(dict(self.config))

            if self.logger is not None:
                self.logger.info(f"Initialized wandb with project {self.config.wandb_project}")
            else:
                print(f"Initialized wandb with project {self.config.wandb_project}")
        else:
            if self.logger is not None:
                self.logger.info("wandb not initialized")
            else:
                print("wandb not initialized")

    def log(self, metric, t):
        wandb.log(metric, step=t)


# CSV logging handler
class CSVLoggingHandler(BaseLoggingHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log(self, metric, t):


        # Load or create a CSV dataframe
        csv_path = os.path.join(self.config.exp_dir, "progress.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame()

        metric['timestep'] = t

        # Append new data
        new_data = pd.DataFrame.from_dict({k: [v] for k, v in metric.items()})
        df = pd.concat([df, new_data], ignore_index=True)

        # Save the updated CSV
        df.to_csv(csv_path, index=False)


# Multiple logging handler
class MultipleLoggingHandler:
    def __init__(self, handler_classes: list, **kwargs):
        self.handlers = [handler_class(**kwargs) for handler_class in handler_classes]

    def set_start_time(self, start_time):
        for handler in self.handlers:
            handler.set_start_time(start_time)

    def set_steps_prev_complete(self, steps):
        for handler in self.handlers:
            handler.set_steps_prev_complete(steps)

    def log(self, metric, t):
        for handler in self.handlers:
            handler.log(metric, t)
