import os
import json
import rapid_attention as RA


project_root = RA.utils.common.find_project_root()
datasets_dir = project_root / "datasets"

def fetch_raw_jsonl_lines(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line

def clean_deepctrl_dataset_for_pretrain():
    """数据集包含字段:
        1.id，用于追踪数据的唯一标识符。
        2.instruction，系统提示词。
        3.input，用户输入指令
        4.output，输出
        5.history，历史对话
        6.language，语言
        7.data_source，数据来源
        8.input_len，用户平均单轮输入长度
        9.output_len，平均输出长度
        10.num_utter，对话轮次
        11.type，数据类别
        12.type_keyword，该类别数据的关键词
    """
    raw_dataset_path = project_root / "datasets" / "deepctrl" / "deepctrl-sft-data" / "sft_data_zh.jsonl"
    cleaned_dataset_path = project_root / "datasets" / "deepctrl" / "deepctrl-sft-data" / "pretrain_data_zh_clean.jsonl"

    print(f"清理数据集 {raw_dataset_path}，保存到: {cleaned_dataset_path}")
    with open(cleaned_dataset_path, "w", encoding="utf-8") as out_f:
        for line in fetch_raw_jsonl_lines(raw_dataset_path):
            item = json.loads(line)
            text = f"{item.get("input", "").strip()} {item.get("output", "").strip()}"
            history = item.get("history")
            for history_item in history:
                if history_item and len(history_item) == 2:
                    text += f" {history_item[0].strip()} {history_item[1].strip()}"
            if text:
                cleaned_item = {"text": text}
                out_f.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
    print("数据集清理完成。")


def clean_deepctrl_dataset_for_sft():
    """数据集包含字段:
        1.id，用于追踪数据的唯一标识符。
        2.instruction，系统提示词。
        3.input，用户输入指令
        4.output，输出
        5.history，历史对话
        6.language，语言
        7.data_source，数据来源
        8.input_len，用户平均单轮输入长度
        9.output_len，平均输出长度
        10.num_utter，对话轮次
        11.type，数据类别
        12.type_keyword，该类别数据的关键词
    """
    raw_dataset_path = project_root / "datasets" / "deepctrl" / "deepctrl-sft-data" / "sft_data_zh.jsonl"
    cleaned_dataset_path = project_root / "datasets" / "deepctrl" / "deepctrl-sft-data" / "sft_data_zh_clean.jsonl"

    print(f"清理数据集 {raw_dataset_path}，保存到: {cleaned_dataset_path}")
    with open(cleaned_dataset_path, "w", encoding="utf-8") as out_f:
        for line in fetch_raw_jsonl_lines(raw_dataset_path):
            item = json.loads(line)
            # 提取对话内容, 增加特殊标记
            # <|im_start|> 用于标记对话开始
            # <|im_end|> 用于标记对话结束
            text = f"<|im_start|> {item.get("input", "").strip()} <|im_end|> <|im_start|>{item.get("output", "").strip()}  <|im_end|>"
            if text:
                cleaned_item = {"text": text}
                out_f.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
    print("数据集清理完成。")

def clean_stepfun_ai_dataset_for_pretrain():
    """
    每个原始分片是一个 JSON 文件，其顶层为示例列表。每个示例目前包含一个 conversations 字段，其中按顺序记录了多轮对话消息。
{
  "conversations": [
    {
      "role": "user",
      "content": "...",
      "loss_mask": 1,
      "name": "",
      "meta": {}
    },
    {
      "role": "assistant",
      "content": "...",
      "loss_mask": 1,
      "name": "",
      "meta": {},
      "reasoning_content": "..."
    }
  ]
}
已观察到的字段包括：
role：发言者角色，包括 user assistant tool system
content：可见的消息文本
loss_mask：轮次级别的监督标志
name：可选的发言者名称
meta：每轮对话的元数据
reasoning_content：在某些示例中出现的、助理端可选字段
    """
    raw_dataset_path = project_root / "datasets" / "stepfun-ai" / "Step-3.5-Flash-SFT" / "json" / "general"
    cleaned_dataset_path = project_root / "datasets" / "stepfun-ai" / "Step-3.5-Flash-SFT" / "pretrain_data_clean.jsonl"
    # 使用正则表达式选择符合条件的文件，部分读取样本数据 chunk_0.json ~ chunk_20.json 读取前 20 个分片文件
    for json_file in [raw_dataset_path / f"chunk_{i}.json" for i in range(20)]:
        with open(json_file, "r", encoding="utf-8") as f:
            coversations = json.load(f)
            with open(cleaned_dataset_path, "a", encoding="utf-8") as out_f:
                for conversation in coversations:
                    text = ""
                    conversation_obj = conversation.get("conversations", {})
                    for message in conversation_obj:
                        message_role = message.get("role")
                        if message_role is not None and message_role in ["user", "assistant", "system"]:
                            text += f"{message.get("content", "").strip()} "
                    if text:
                        cleaned_item = {"text": text.strip()}
                        out_f.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")


def clean_stepfun_ai_dataset_for_sft_train():
    raw_dataset_path = project_root / "datasets" / "stepfun-ai" / "Step-3.5-Flash-SFT" / "json" / "general"
    cleaned_dataset_path = project_root / "datasets" / "stepfun-ai" / "Step-3.5-Flash-SFT" / "sft_data_clean.jsonl"
    # 使用正则表达式选择符合条件的文件，部分读取样本数据 chunk_0.json ~ chunk_20.json 读取前 20 个分片文件
    for json_file in [raw_dataset_path / f"chunk_{i}.json" for i in range(20)]:
        with open(json_file, "r", encoding="utf-8") as f:
            coversations = json.load(f)
            # 把普通的json文件转换成jsonl格式，每行一个json对象，方便后续处理
            with open(cleaned_dataset_path, "a", encoding="utf-8") as out_f:
                for conversation in coversations:
                    out_f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # clean_deepctrl_dataset_for_pretrain()
    # clean_deepctrl_dataset_for_sft()
    # clean_stepfun_ai_dataset_for_pretrain()
    clean_stepfun_ai_dataset_for_sft_train()