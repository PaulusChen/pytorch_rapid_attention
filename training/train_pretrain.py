import torch
import rapid_attention
from pathlib import Path
from utils.logger import Logger
from transformers import AutoTokenizer
from contextlib import nullcontext
from transformers import PretrainedConfig
from rapid_attention.utils.global_context import (
    rapid_attention_global_context as GCTX,
)
from model.model_rapid_attention import RapidAttentionLMConfig
from model.model_rapid_attention import RapidAttentionForCausalLM


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = self.load_data(data_path)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        X = input_ids[:-1].long()
        Y = input_ids[1:].long()
        loss_mask = loss_mask[1:].bool()
        return X, Y, loss_mask


    def load_data(self, data_path):
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                samples.append(json.loads(line))
        return samples


def main():
    lm_config = RapidAttentionLMConfig()
    tokenizer_model_path = Path(GCTX.tokenizer_config.tokenizer_dir)
    ctx = nullcontext() if GCTX.common_config.device_type == "cpu" else torch.cuda.amp.autocast()

    device = f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else "cpu"

    base_seed = GCTX.common_config.seed
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if GCTX.train_config.use_wandb:
        import wandb
        wandb.init(
            project=GCTX.train_config.wandb_project,
            name=lm_config.wandb_run_name
        )
    else:
        wandb = None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    model = RapidAttentionForCausalLM(lm_config).to(device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    trainn_ds = 


if __name__ == "__main__":
    main()
