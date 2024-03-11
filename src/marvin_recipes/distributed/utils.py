import os

import torch

from marvin_recipes.misc.utils import set_seed, is_bf16_supported
from torch import distributed as dist
import logging

logger = logging.getLogger(__name__)


def get_process_info():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return local_rank, rank, world_size


def log_init(local_rank, rank, world_size):
    if local_rank == 0:
        logger.info(
            f"Dist init done, local_rank {local_rank}, rank {rank}, world_size {world_size}, Current device {torch.cuda.current_device()}")
        if is_bf16_supported():
            logger.info("BF16 is supported")
        else:
            logger.warning("BF16 is NOT supported")


def setup_distributed_process(seed, dist_backend):
    set_seed(seed)
    dist.init_process_group(dist_backend)
    local_rank, rank, world_size = get_process_info()
    log_init(local_rank, rank, world_size)

    if torch.distributed.is_initialized():
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
    else:
        logger.error("Distributed is not initialized")
        raise RuntimeError("Distributed is not initialized")

    return local_rank, rank, world_size
