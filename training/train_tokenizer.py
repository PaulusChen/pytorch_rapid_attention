import os
import json
import random
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
import rapid_attention as RA
from rapid_attention.utils.global_context import (
    rapid_attention_global_context as RAP_GCTX,
)


def tokenizer_texts_iter(data_path):
    print(f"读取数据文件: {data_path}")
    train_lines = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            # 只采样 10% 数据可以明显加快试验速度，
            # 但也会降低 tokenizer 对长尾词和罕见字符的覆盖能力。
            # TODO: 当你开始认真比较模型效果时，可以自己把采样比例调高，例如：
            # TODO: 示例改法
            # sample_ratio = 0.5
            # if random.random() > sample_ratio:
            #     continue
            if random.random() > 0.1:
                continue
            train_lines += 1
            try:
                item = json.loads(line)
                text = item.get("text", "")
                if not text:
                    continue
                yield text
            except json.JSONDecodeError:
                print(f"跳过无效行: {line}")
                continue
    print(f"训练数据总行数: {train_lines}")


def train_tokenizer(data_path, tokenizer_dir, vocab_size):
    print(f"训练分词器... 词汇表大小: {vocab_size}")
    tokenizer = Tokenizer(models.BPE())
    # ByteLevel BPE 对任意字节序列都有定义，
    # 对中英文混合文本、符号和未知字符通常更稳健。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
        ],
        show_process=True,
        max_memory={0: "40GB"},
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    texts_iter = tokenizer_texts_iter(data_path)
    tokenizer.train_from_iterator(texts_iter, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    # 这里断言特殊 token 的 id，是为了让 tokenizer 产物和后续模型配置中的
    # bos/eos/pad id 保持严格一致。
    assert tokenizer.token_to_id("<|endoftext|>") == 0, "确保<|endoftext|>的ID为0"
    assert tokenizer.token_to_id("<|im_start|>") == 1, "确保<|im_start|>的ID为1"
    assert tokenizer.token_to_id("<|im_end|>") == 2, "确保<|im_end|>的ID为2"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(str(tokenizer_dir))
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            },
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": False,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        # TODO: 这个 chat_template 只有在推理代码里显式调用
        # `tokenizer.apply_chat_template(...)` 时才会真正生效。
        # 如果推理脚本只是简单地拼 `bos_token + prompt`，那么这里的模板
        # 和多轮对话格式实际上不会参与模型输入。
        # https://huggingface.co/docs/transformers/v5.3.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
        # https://huggingface.co/docs/transformers/v5.4.0/zh/chat_templating
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}"
    }
    with open(
        os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"分词器训练完成并保存在: {tokenizer_dir}")


if __name__ == "__main__":
    if not os.path.exists(RAP_GCTX.tokenizer_config.tokenizer_dir):
        os.makedirs(RAP_GCTX.tokenizer_config.tokenizer_dir)
    train_tokenizer(
        data_path=RAP_GCTX.tokenizer_config.tokenizer_train_data_path,
        tokenizer_dir=RAP_GCTX.tokenizer_config.tokenizer_dir,
        vocab_size=RAP_GCTX.tokenizer_config.vocab_size,
    )
