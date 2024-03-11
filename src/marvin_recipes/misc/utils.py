import packaging
import torch
from torch import distributed as dist
from torch.cuda import nccl
import random
import numpy as np
import logging

from marvin_recipes.models.model_definitions import LanguageModel

logger = logging.getLogger(__name__)


def is_bf16_supported():
    return (torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and packaging.version.parse(torch.version.cuda).release >= (11, 0)
            and dist.is_nccl_available()
            and nccl.version() >= (2, 10)
            )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_model_size(model, local_rank=0, run_name=None):
    if local_rank != 0:
        return
    logger.info(f"--> Model {model.__class__}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n--> {model.__class__} has {total_params / 1e6} Million params\n")
    if run_name:
        logger.info(f"--> Loaded model for Run: {run_name}\n")


def log_train_start(local_rank, data_module, l_model: LanguageModel):
    if local_rank != 0:
        return
    logger.info(f"Complete Training Set Length = {data_module.train_dataset.num_samples}")
    logger.info(f"Data samples per GPU (in actual shard) = {len(data_module.train_dataset)}")
    logger.info(f"### Model size after sharding ###")
    print_model_size(l_model.model, local_rank)