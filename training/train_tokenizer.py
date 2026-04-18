import os
import json
import random
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
import rapid_attention as RA
from rapid_attention.utils.global_context import (
    rapid_attention_global_context as RAP_GCTX,
)

VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36

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


def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM):
    print(f"训练分词器... 词汇表大小: {vocab_size}")

    # 设定特殊 tokens 列表，包含多模态相关的 tokens 和一些预留的 buffer tokens。
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", 
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", 
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", 
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    # 这里进行tokens的分层设计，第一层是基础的特殊tokens，第二层是工具调用相关的, 这部分只在推理阶段进行生成
    # !!! 为什么要这么处理 Special token 本质是"协议标记" —— 它们定义了模型如何理解结构，但不是给用户看的内容。
    # 所以 tokenizer在decode时会跳过  Special token 
    """
    # 编码: 文本 → token IDs
    tokens = tokenizer.encode("<|endoftext|>Hello world")
    # tokens: [50256, 15496, 995] 
    # 50256 is the ID for <|endoftext|>

    # 解码: token IDs → 文本
    text = tokenizer.decode(tokens)

    # 输出: "Hello world" 或 "<|endoftext|>Hello world"
    # 取决于 special 字段:
    #   - 如果 special=false: 输出包含 "<|endoftext|>"
    #   - 如果 special=true: 跳过该token，只输出 "Hello world"
    # 所以下面这些需要客户端处理的情况, token的special字段都应该设置为 False, 这样在 decode 时就不会被 tokenizer 跳过。
    
    """
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    ]
    num_buffer = special_tokens_num - len(special_tokens_list) - len(additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)] # 预留一定数量的token位置
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
    tokenizer = Tokenizer(models.BPE())
    # ByteLevel BPE 对任意字节序列都有定义，
    # 对中英文混合文本、符号和未知字符通常更稳健。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=all_special_tokens,
        show_process=True,
        max_memory={0: "40GB"},
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    texts_iter = tokenizer_texts_iter(data_path)
    tokenizer.train_from_iterator(texts_iter, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_tokens_list)
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(str(tokenizer_dir))

    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    for token_info in tokenizer_data.get('added_tokens', []):
        # 这里识别出 special_tokens_list 中的 token，并标记它们为 special=True，其他的 buffer tokens 则标记为 special=False。
        # 也就是第一层的特殊 tokens 会被 tokenizer 识别为真正的特殊 tokens，而第二层的工具调用相关 tokens 和 buffer tokens 则不会被 tokenizer 识别为特殊 tokens，
        # 这样在模型训练和推理时就可以区分对待它们。
        if token_info['content'] not in special_tokens_list:
            token_info['special'] = False
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    added_tokens_decoder = {}
    for i, token in enumerate(all_special_tokens):
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True if token in special_tokens_list else False
        }
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [t for t in special_tokens_list if t not in ["<|endoftext|>"]],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        # TODO: 这个 chat_template 只有在推理代码里显式调用
        # `tokenizer.apply_chat_template(...)` 时才会真正生效。
        # 如果推理脚本只是简单地拼 `bos_token + prompt`，那么这里的模板
        # 和多轮对话格式实际上不会参与模型输入。
        # https://huggingface.co/docs/transformers/v5.3.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
        # https://huggingface.co/docs/transformers/v5.4.0/zh/chat_templating
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast"
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
