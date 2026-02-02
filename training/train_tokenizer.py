import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
import rapid_attention as RA


project_root = RA.utils.common.find_project_root()
data_dir = project_root / "datasets" / "deepctrl" / "deepctrl-sft-data" / "sft_data_zh.json"
vocab_size = 6400

def train_tokenizer(data_dir, vocab_size=vocab_size, special_tokens=None, tokenizer_save_path="tokenizer.json"):
    print(f"训练分词器... 词汇表大小: {vocab_size}")
    pass


if __name__ == "__main__":
    train_tokenizer(data_dir)
