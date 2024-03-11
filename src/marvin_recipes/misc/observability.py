from abc import ABC
from typing import Literal
import wandb


class TrainingTracker(ABC):
    """Abstract base class for training trackers."""

    def __init__(self, project_name: str, run_name: str, **kwargs):
        self._project_name = project_name
        self._run_name = run_name
        self._kwargs = kwargs

    def log_metrics(self, metrics: dict, step: int = None):
        pass

    def trace_input(self, inputs: str, labels: str = None, shape: str = None):
        pass

    def add_hparams(self, hparams: dict):
        pass

    def close(self):
        pass


class WandbTracker(TrainingTracker):

    def __init__(self, project_name: str, run_name: str, **kwargs):
        super().__init__(project_name, run_name, **kwargs)
        self.run = wandb.init(project=project_name, name=run_name, **kwargs)
        # self.run = wandb.init(project=project_name, **kwargs)
        self.traces_count = 0
        self.text_table = wandb.Table(columns=["inputs", "labels", "shape"])

    def log_metrics(self, metrics: dict, step: int = None):
        wandb.log(metrics, step=step)

    def trace_input(self, inputs: str, labels: str = None, shape: str = None):
        self.text_table.add_data(inputs, labels, shape)
        self.traces_count += 1
        if self.traces_count % 2 == 0:
            wandb.log({"inputs_data": self.text_table})

    def add_hparams(self, hparams: dict):
        wandb.config.update(hparams)

    def close(self):
        wandb.log({"inputs_data": self.text_table})
        wandb.finish()


class NoOpTracker(TrainingTracker):

    def __init__(self, project_name: str, run_name: str, **kwargs):
        super().__init__(project_name, run_name, **kwargs)


def get_tracker(provider: Literal["mlflow", "wandb", None], project_name: str, run_name: str, rank: int, local_rank: int,
                **kwargs) -> TrainingTracker:
    """Get the tracker instance for the given provider."""
    if local_rank != 0 or provider is None:
        print("Not logging metrics, rank is not 0 or provider is None")
        return NoOpTracker(project_name, run_name, **kwargs)
    if provider == "wandb":
        return WandbTracker(project_name, f"{run_name}_r{rank}", group=f"{run_name}_group", **kwargs)
    else:
        raise ValueError(f"Unsupported tracker provider: {provider}")
