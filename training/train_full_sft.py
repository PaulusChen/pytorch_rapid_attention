from pandas.io import json
import torch

import time
from rapid_attention import logger
from rapid_attention.utils.global_context import (
    rapid_attention_global_context as GCTX,
)
from model.model_rapid_attention import RapidAttentionLMTrainerContext
from model.model_rapid_attention import RapidAttentionForCausalLM
from torch.utils.data import Dataset
from datasets import load_dataset, Features, Value


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features(
            {
                "conversations": [
                    {
                        "role": Value("string"),
                        "content": Value("string"),
                        "reasoning_content": Value("string"),
                        "tools": Value("string"),
                        "tool_calls": Value("string"),
                    }
                ]
            }
        )
        self.samples = load_dataset(
            "json", data_files=str(jsonl_path), features=features
        )["train"]
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = (
                    json.loads(message["tools"])
                    if isinstance(message["tools"], str)
                    else message["tools"]
                )
            if message.get("tool_calls"):
                message["tool_calls"] = (
                    json.loads(message["tool_calls"])
                    if isinstance(message["tool_calls"], str)
                    else message["tool_calls"]
                )
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels


def main():
    # 初始化训练上下文，包括配置、日志和 tokenizer。
    trainer_ctx = RapidAttentionLMTrainerContext(stage="full_sft")
    # 设置混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # 定义模型、优化器
    model = RapidAttentionForCausalLM(trainer_ctx.lm_config)

    # 构建训练数据集和数据加载器。
    train_ds = SFTDataset(
        GCTX.train_config.sft_train_data_path,
        trainer_ctx.tokenizer,
        max_len=trainer_ctx.lm_config.max_seq_len,
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
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(GCTX.train_config.dtype in ["float16", "bfloat16"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=GCTX.train_config.learning_rate
    )

    start_epoch = 0
    start_step = 0
    checkpoint = GCTX.train_config.sft_checkpoint_pth
    if not checkpoint.exists():
        checkpoint = GCTX.train_config.pretrain_checkpoint_pth
    if checkpoint.exists():
        ckp_data = torch.load(checkpoint, map_location=trainer_ctx.device)
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        logger(f"从 {checkpoint} 恢复模型参数成功。")
    else:
        logger(f"未找到预训练模型，开始从头训练。")

    model = torch.compile(model)
    Logger("torch.compile 加速已启用")
    iter_per_epoch = len(train_loader)
    for epoch in range(start_epoch, GCTX.train_config.epochs):
        start_time = time.time()
        last_step = start_step
        for step, (input_ids, labels) in enumerate(train_loader, start=start_step + 1):
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
            with autocast_ctx:
                res = model(input_ids, labels)
                loss = res.loss + res.aux_loss
                loss = loss / GCTX.train_config.accumulation_steps
            scaler.scale(loss).backward()
            if step % GCTX.train_config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_clip = 1.0  # 可以根据需要调整梯度裁剪的阈值
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % GCTX.train_config.log_interval == 0 or step == iter_per_epoch:
                spend_time = time.time() - start_time
                current_loss = loss.item() * GCTX.train_config.accumulation_steps
                current_aux_loss = (
                    res.aux_loss.item() if res.aux_loss is not None else 0.0
                )
                current_logits_loss = current_loss - current_aux_loss
                current_lr = optimizer.param_groups[-1]["lr"]
                eta_min = (
                    spend_time
                    / max(step - last_step, 1)
                    * (iter_per_epoch - step)
                    // 60
                )
                logger(
                    f"Epoch [{epoch+1}/{GCTX.train_config.epochs}] Step [{step}/{iter_per_epoch}] "
                    f"Loss: {current_loss:.4f} (Aux: {current_aux_loss:.4f}, Logits: {current_logits_loss:.4f}) "
                    f"LR: {current_lr:.2e} ETA: {eta_min:.2f} mins"
                )
                if trainer_ctx.wandb is not None:
                    trainer_ctx.wandb.log(
                        {
                            "loss": current_loss,
                            "logits_loss": current_logits_loss,
                            "aux_loss": current_aux_loss,
                            "lr": current_lr,
                            "eta": eta_min,
                        }
                    )
            if step % GCTX.train_config.save_interval == 0 or step == iter_per_epoch:
                model.eval()
                ckp_pth = GCTX.train_config.sft_checkpoint_pth
                tmp_ckp_pth = ckp_pth.with_suffix(".tmp")
                state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "step": step,
                }
                torch.save(
                    {k: v.half().cpu() for k, v in state_dict.items()}, tmp_ckp_pth
                )
                tmp_ckp_pth.rename(ckp_pth)
                model.train()
                del state_dict
            del input_ids, labels, res, loss


if __name__ == "__main__":
    main()
