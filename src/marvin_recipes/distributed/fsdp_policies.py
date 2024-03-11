import torch
from functools import partial
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from marvin_recipes.models.model_definitions import LanguageModel


def get_policies_for_model(l_model: LanguageModel, bf16_enabled):
    if not bf16_enabled:
        raise ValueError("training is only supported with bf16 enabled for now.")
    mixed_p = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16)
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={l_model.transformer_block, })

    return {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_p
    }
