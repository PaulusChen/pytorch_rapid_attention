
import argparse
import json
from transformers import PretrainedConfig
from rapid_attention.utils.global_context import rapid_attention_global_context as RAP_GCTX


class RapidAttentionConfig(PretrainedConfig):
    model_type = "rapid_attention"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size", RAP_GCTX.vocab_size)
        common_eps: float = kwargs.get("common_eps", 1e-6)


def main():
    parser = argparse.ArgumentParser(description="Pretrain a Rapid Attention model.")
    parser.add_argument("--config_file", type=str, help="Path to the model configuration file.")
    args = parser.parse_args()
    config = json.load(open(args.config_file, "r"))
    lm_config = RapidAttentionConfig(**config)
    



if __name__ == "__main__":
    main()