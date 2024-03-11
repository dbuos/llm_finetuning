from dataclasses import dataclass
from transformers import PreTrainedModel


@dataclass
class LanguageModel:
    model: PreTrainedModel
    transformer_block: type
    activation_chk: callable


def init_llama_model(model):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    return LanguageModel(
        model=model,
        transformer_block=LlamaDecoderLayer,
        activation_chk=lambda sb: isinstance(sb, LlamaDecoderLayer),
    )


def init_mixtral_model(model):
    from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
    return LanguageModel(
        model=model,
        transformer_block=MixtralDecoderLayer,
        activation_chk=lambda sb: isinstance(sb, MixtralDecoderLayer),
    )


def init_qwen2_model(model):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    return LanguageModel(
        model=model,
        transformer_block=Qwen2DecoderLayer,
        activation_chk=lambda sb: isinstance(sb, Qwen2DecoderLayer),
    )
