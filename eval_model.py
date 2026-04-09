import torch
import random
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from rapid_attention.utils.global_context import (
    rapid_attention_global_context as GCTX,
)

from model.model_rapid_attention import RapidAttentionLMConfig
from model.model_rapid_attention import RapidAttentionForCausalLM


def setup_seed(seed):
    # 推理阶段固定随机种子，主要是为了让采样结果更容易复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_module_path(type):
    module_path = Path(GCTX.train_config.output_dir)
    return module_path / f"{type}_{GCTX.model_config.hidden_size}.pth"


def main():
    tokenizer_model_path = Path(GCTX.tokenizer_config.tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)

    lm_config = RapidAttentionLMConfig()
    model = RapidAttentionForCausalLM(lm_config)

    # TODO: 这里没有显式指定 `map_location`。
    # 如果 checkpoint 保存设备与当前加载设备不一致，可能带来问题。
    # 你可以自己尝试改成：
    # TODO: 示例改法
    # model.load_state_dict(torch.load(get_module_path("pretrain"), map_location=GCTX.common_config.device))
    model.load_state_dict(torch.load(get_module_path("pretrain")))
    model = model.eval().to(GCTX.common_config.device)

    prompt_datas = [
        '请介绍一下自己。',
        '你更擅长哪一个学科？',
        '鲁迅的《狂人日记》是如何批判封建礼教的？',
        '我咳嗽已经持续了两周，需要去医院检查吗？',
        '详细的介绍光速的物理概念。',
        '推荐一些杭州的特色美食吧。',
        '请为我讲解“大语言模型”这个概念。',
        '如何理解ChatGPT？',
        'Introduce the history of the United States, please.'
    ]
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    messages = []
    for idx, prompt in enumerate(prompt_datas if test_mode == 0 else iter(lambda: input('🧑‍💼：'), '')):
        # TODO: 这里每轮都重新随机设置 seed，会让“可复现”和“随机性”
        # 混在一起。你可以自己明确选择一种策略，例如：
        # TODO: 示例改法 1（固定复现）
        # setup_seed(GCTX.common_config.seed)
        # TODO: 示例改法 2（同一轮次可复现）
        # setup_seed(GCTX.common_config.seed + idx)
        setup_seed(random.randint(0, 1000))
        # TODO: 这里打印的是用户 prompt，但前缀写成了“🤖”。
        # 你可以自己改成：
        # TODO: 示例改法
        # if test_mode == 0:
        #     print(f'🧑‍💼：{prompt}')
        if test_mode == 0: print(f'🤖：{prompt}')

        messages = messages[-GCTX.eval_config.max_history_len:]
        messages.append({"role": "user", "content": prompt})
        '''
        new_prompt = tokenizer.apply_chat_template(messages,
                                                   tokenize=False,
                                                   add_generation_prompt=True
                                                  )'''
        new_prompt = tokenizer.bos_token + prompt
        # TODO: 这里虽然维护了 `messages`，但真正送进模型的只有 `bos_token + prompt`，
        # 多轮历史实际上没有参与推理。
        # 你可以自己把上面的 `apply_chat_template` 接回来，例如：
        # TODO: 示例改法
        # new_prompt = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(GCTX.common_config.device)

        print('🤖: ', end='')
        generated_ids = model.generate(
            inputs["input_ids"],
            # TODO: 当前配置里 `max_new_tokens=8192`，但模型配置的
            # `max_position_embeddings=512`。如果总长度超过位置编码上限，
            # 推理会出现越界或形状不匹配风险。
            # 你可以自己先做一个保护，例如：
            # TODO: 示例改法
            # max_context_left = lm_config.max_position_embeddings - inputs["input_ids"].shape[1]
            # max_new_tokens = min(GCTX.eval_config.max_new_tokens, max_context_left)
            max_new_tokens=GCTX.eval_config.max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=GCTX.eval_config.top_p,
            temperature=GCTX.eval_config.temperature
        )
        response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        print('\n\n')


if __name__ == "__main__":
    main()
