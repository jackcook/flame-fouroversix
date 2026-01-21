from fla.models.transformer import TransformerConfig
from fouroversix import AdaptiveBlockScalingRule, QuantizeBackend


class FP4TransformerConfig(TransformerConfig):

    model_type = "fp4_transformer"

    def __init__(self, layer_precision_configs: list[dict] | None = None, **kwargs):
        self.layer_precision_configs = (
            [
                {
                    k: (
                        AdaptiveBlockScalingRule(v)
                        if k == "scale_rule"
                        else (QuantizeBackend(v) if k == "quantize_backend" else v)
                    )
                    for k, v in config.items()
                }
                for config in layer_precision_configs
                for _ in range(config.pop("repeats", 1))
            ]
            if layer_precision_configs is not None
            else None
        )

        super().__init__(**kwargs)
