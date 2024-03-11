from functools import partial
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)

from marvin_recipes.distributed.fsdp_policies import get_policies_for_model
from marvin_recipes.misc.utils import is_bf16_supported
from marvin_recipes.models.model_definitions import LanguageModel
import logging

logger = logging.getLogger(__name__)


def apply_checkpointing(l_model: LanguageModel):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing
    )
    _wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        l_model.model, checkpoint_wrapper_fn=_wrapper, check_fn=l_model.activation_chk
    )


def shard_and_distribute(l_model: LanguageModel, activation_checkpointing=False, local_rank=0):
    policies = get_policies_for_model(l_model, is_bf16_supported())
    model = FSDP(
        l_model.model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        **policies
    )

    if activation_checkpointing:
        apply_checkpointing(l_model)
        if local_rank == 0:
            logger.info(f"Applying FSDP activation checkpointing to model {l_model.model.__class__}")

    l_model.model = model
    return l_model
