import os
import torch
import time
import json
import math
from rapid_attention import logger
from rapid_attention.utils.global_context import (
    rapid_attention_global_context as GCTX,
)
from model.model_rapid_attention import RapidAttentionLMTrainerContext
from model.model_rapid_attention import RapidAttentionForCausalLM


def get_lr(current_step, total_steps, lr):
    # 这是一个余弦退火风格的学习率调度。
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
        # 这里假定每条样本都有 `text` 字段，并将其编码成定长 token 序列。
        # 对预训练来说，本质上就是做 next-token prediction。
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_len - 2,
            truncation=True
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # -100 是 CrossEntropyLoss 的 ignore_index
        return input_ids, labels

    def load_data(self, data_path):
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                samples.append(json.loads(line))
        return samples


def main():
    # 初始化训练上下文，包括配置、日志和 tokenizer。
    trainer_ctx = RapidAttentionLMTrainerContext(stage="pretrain")

    # 构建模型
    model = RapidAttentionForCausalLM(trainer_ctx.lm_config).to(trainer_ctx.device)
    logger(
        f"LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )

    # 构建训练数据集和数据加载器。
    train_ds = PretrainDataset(
        GCTX.train_config.pre_train_data_path, trainer_ctx.tokenizer, max_len=trainer_ctx.lm_config.max_seq_len
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=GCTX.train_config.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=(train_sampler is None),
        num_workers=GCTX.train_config.num_workers,
        sampler=train_sampler,
    )

    # 构建优化器和学习率调度器。这里使用了混合精度训练和梯度累积来提升训练效率。
    scaler = torch.amp.GradScaler("cuda",
        enabled=(GCTX.train_config.dtype in ["float16", "bfloat16"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=GCTX.train_config.learning_rate
    )
    
    # 从 ckpt 恢复训练状态（如果有的话）。
    start_epoch = 0
    start_step = 0
    checkpoint = GCTX.train_config.pretrain_checkpoint_pth
    if checkpoint.exists():
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scaler.load_state_dict(state_dict["scaler"])
        start_epoch = state_dict.get("epoch", 0)
        start_step = state_dict.get("step", 0)
        logger(f"从 {checkpoint} 恢复模型参数成功。")
    iter_per_epoch = len(train_loader)
    for epoch in range(start_epoch, GCTX.train_config.epochs):
        start_time = time.time()
        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(trainer_ctx.device)
            labels = labels.to(trainer_ctx.device)
            last_step = step

            lr = get_lr(
                epoch * iter_per_epoch + step,
                GCTX.train_config.epochs * iter_per_epoch,
                GCTX.train_config.learning_rate,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with trainer_ctx.ctx:
                res = model(input_ids, labels=labels)
                loss = res.loss + res.aux_loss
                loss = loss / GCTX.train_config.accumulation_steps

            scaler.scale(loss).backward()

            if step % GCTX.train_config.accumulation_steps == 0:
                # 梯度累积：多个 micro-batch 共用一次参数更新。
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GCTX.train_config.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % GCTX.train_config.log_interval == 0 or step == iter_per_epoch:
                spend_time = time.time() - start_time
                current_loss = loss.item() * GCTX.train_config.accumulation_steps
                current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
                current_logits_loss = current_loss - current_aux_loss
                current_lr = optimizer.param_groups[-1]["lr"]
                eta_min = spend_time / max(step - start_step, 1) * (iter_per_epoch - step) // 60
                logger(
                    f"Epoch: {epoch} / {GCTX.train_config.epochs}, Step: {step}, IterPerEpoch: {iter_per_epoch}, Loss: {loss.item():.4f}, current_lr: {current_lr:.2e}, ETA: {eta_min:.0f}min"
                )
                if trainer_ctx.wandb is not None:
                    trainer_ctx.wandb.log(
                        {
                            "train_loss": loss.item(),
                            "train_aux_loss": current_aux_loss,
                            "train_logits_loss": current_logits_loss,
                            "learning_rate": optimizer.param_groups[-1]["lr"],
                            "epoch": epoch,
                            "step": step,
                        }
                    )
            if step % GCTX.train_config.save_interval == 0 or step == iter_per_epoch:
                model.eval()
                checkpoint_pth = GCTX.train_config.pretrain_checkpoint_pth
                if not os.path.exists(checkpoint_pth.parent):
                    os.makedirs(checkpoint_pth.parent, exist_ok=True)
                state_dict = model.state_dict()
                state_dict = {k: v.cpu() for k, v in state_dict.items()}
                resume_data = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "step": step,
                }
                checkpoint_tmp_path = checkpoint_pth.with_suffix(".tmp")
                torch.save(resume_data, checkpoint_tmp_path)
                os.replace(checkpoint_tmp_path, checkpoint_pth)
                model.train()
                del state_dict
            del input_ids, labels, res, loss
        if last_step > start_step and last_step % GCTX.train_config.accumulation_steps != 0:
            # 处理 epoch 末尾剩余的梯度更新。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GCTX.train_config.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
