from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_transformer import FP4TransformerConfig
from .modeling_transformer import (
    FP4TransformerForCausalLM,
    FP4TransformerModel,
)

AutoConfig.register(
    FP4TransformerConfig.model_type, FP4TransformerConfig, exist_ok=True,
)
AutoModel.register(FP4TransformerConfig, FP4TransformerModel, exist_ok=True)
AutoModelForCausalLM.register(
    FP4TransformerConfig, FP4TransformerForCausalLM, exist_ok=True,
)


__all__ = ["FP4TransformerConfig", "FP4TransformerForCausalLM", "FP4TransformerModel"]
