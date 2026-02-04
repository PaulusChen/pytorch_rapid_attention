import os
import json
import rapid_attention as RA


project_root = RA.utils.common.find_project_root()
datasets_dir = project_root / "datasets"

def fetch_raw_jsonl_lines(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line

def clean_deepctrl_dataset(dataset_dir):
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




if __name__ == "__main__":
    clean_deepctrl_dataset(datasets_dir)