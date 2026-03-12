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
        setup_seed(random.randint(0, 1000))
        if test_mode == 0: print(f'🤖：{prompt}')

        messages = messages[-GCTX.eval_config.max_history_len:]
        messages.append({"role": "user", "content": prompt})
        '''
        new_prompt = tokenizer.apply_chat_template(messages,
                                                   tokenize=False,
                                                   add_generation_prompt=True
                                                  )'''
        new_prompt = tokenizer.bos_token + prompt
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(GCTX.common_config.device)

        print('🤖: ', end='')
        generated_ids = model.generate(
            inputs["input_ids"],
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