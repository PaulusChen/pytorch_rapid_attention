from torch.utils.data import DistributedSampler
import torch
import time
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


def Logger(content):
    print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


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
            str(sample["text"]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = input_ids != self.tokenizer.pad_token_id
        X = input_ids[:-1].long()
        Y = input_ids[1:].long()
        loss_mask = loss_mask[1:].bool()
        return X, Y, loss_mask

    def load_data(self, data_path):
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                samples.append(json.loads(line))
        return samples


def main():
    lm_config = RapidAttentionLMConfig()
    tokenizer_model_path = Path(GCTX.tokenizer_config.tokenizer_dir)
    ctx = (
        nullcontext()
        if GCTX.common_config.device_type == "cpu"
        else torch.cuda.amp.autocast()
    )

    device = GCTX.common_config.device if torch.cuda.is_available() else "cpu"

    base_seed = GCTX.common_config.seed
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if GCTX.train_config.use_wandb:
        import wandb

        wandb.init(
            project=GCTX.train_config.wandb_project, name=lm_config.wandb_run_name
        )
    else:
        wandb = None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    model = RapidAttentionForCausalLM(lm_config).to(device)
    Logger(
        f"LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )
    train_ds = PretrainDataset(
        GCTX.train_config.train_data_path, tokenizer, max_len=lm_config.max_seq_len
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=GCTX.train_config.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=GCTX.train_config.num_workers,
        sampler=train_sampler,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(GCTX.train_config.dtype in ["float16", "bfloat16"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=GCTX.train_config.learning_rate
    )
    iter_per_epoch = len(train_loader)
    for epoch in range(GCTX.train_config.epochs):
        loss_function = torch.nn.CrossEntropyLoss(reduction="none")
        start_time = time.time()
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X = X.to(GCTX.common_config.device)
            Y = Y.to(GCTX.common_config.device)
            loss_mask = loss_mask.to(GCTX.common_config.device)
            lr = get_lr(
                epoch * iter_per_epoch + step,
                GCTX.train_config.epochs * iter_per_epoch,
                GCTX.train_config.learning_rate,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            with ctx:
                res = model(X)
                loss = loss_function(res.view(-1, res.size(-1)), Y.view(-1)).view(
                    Y.size()
                )
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss += res.aux_loss
                loss = loss / GCTX.train_config.accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % GCTX.train_config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GCTX.train_config.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % GCTX.train_config.log_interval == 0:
                spend_time = time.time() - start_time
                Logger(
                    f"Epoch: {epoch} / {GCTX.train_config.epochs}, Step: {step}, IterPerEpoch: {iter_per_epoch}, Loss: {loss.item():.4f}, LR: {lr:.2e}, Time: {spend_time:.2f}s"
                )
                if wandb is not None:
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "learning_rate": optimizer.param_groups[-1]["lr"],
                            "epoch": epoch,
                            "step": step,
                        }
                    )
            if (step + 1) % GCTX.train_config.save_interval == 0:
                model.eval()
                checkpoint = f"{GCTX.train_config.output_dir}/pretrain_{GCTX.train_config.hidden_size}.pth"
                state_dict = model.state_dict()
                torch.save(state_dict, checkpoint)
                model.train()


if __name__ == "__main__":
    main()
