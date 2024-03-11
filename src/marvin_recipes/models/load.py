import torch
from transformers import (
    AutoModelForCausalLM,
    LlamaPreTrainedModel,
    MixtralPreTrainedModel,
    Qwen2PreTrainedModel
)
from marvin_recipes.models.model_definitions import (
    LanguageModel,
    init_llama_model,
    init_mixtral_model,
    init_qwen2_model
)


# TODO: Add support for memory efficient model loading
def load_from_pretrained_chk(checkpoint_path, dtype=torch.bfloat16) -> LanguageModel:
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=dtype, attn_implementation="sdpa")
    if issubclass(model.__class__, LlamaPreTrainedModel):
        l_model = init_llama_model(model)
    elif issubclass(model.__class__, MixtralPreTrainedModel):
        l_model = init_mixtral_model(model)
    elif issubclass(model.__class__, Qwen2PreTrainedModel):
        l_model = init_qwen2_model(model)
    else:
        raise NotImplementedError(f"Model {model.__class__} is not supported yet.")
    return l_model
