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
    use_model_type = "pretrain"

    checkpoint = GCTX.train_config.pretrain_checkpoint_pth
    if checkpoint.exists():
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict["model"])
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
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    conversation = []
    def safe_input(prompt):
        try:
            return input(prompt)
        except EOFError:
            return None
    for idx, prompt in enumerate(prompt_datas if input_mode == 0 else iter(lambda: safe_input('🧑‍💼：'), '')):
        if (prompt is None) or (prompt.strip().lower() in ['exit', 'quit']):
            print("结束对话。")
            break
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f'🧑‍💼：{prompt}')
        conversation = conversation[-GCTX.eval_config.max_history_len:]
        conversation.append({"role": "user", "content": prompt})
        if use_model_type == "pretrain":
            new_prompt = tokenizer.bos_token + prompt
        else:
            new_prompt = tokenizer.apply_chat_template(conversation,
                                                   tokenize=False,
                                                   add_generation_prompt=True)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(GCTX.common_config.device)

        print('🤖: ', end='')
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=GCTX.eval_config.max_new_tokens,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=GCTX.eval_config.top_p,
            temperature=GCTX.eval_config.temperature,
            repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        print('\n\n')


if __name__ == "__main__":
    main()
