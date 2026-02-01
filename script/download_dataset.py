import os
from pathlib import Path
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def find_project_root(marker_file=".rapid_attention_project_root"):
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker_file).exists():
            return parent
        if parent == parent.parent:
            break
    raise FileNotFoundError(
        f"未找到项目根目录标识文件 '{marker_file}'！\n"
        f"请在项目根目录执行：touch {marker_file}"
    )


def download_wikipedia_dataset(datasets_dir: Path):
    datasets_dir.mkdir(exist_ok=True)
    print("📥 开始下载中文维基数据集...")
    try:
        download_path = snapshot_download(
            repo_id="wikimedia/wikipedia",  # 数据集仓库ID
            repo_type="dataset",  # 明确是数据集（不是模型）
            revision="main",  # 分支
            allow_patterns="20231101.zh/",  # 只下载中文维基目录，避免下其他语言
            local_dir=datasets_dir / "zh_wiki",  # 本地保存目录
        )

    except Exception as e:
        print(f"❌ 下载出错：{str(e)}")

def check_login_huggingface():
    from huggingface_hub import login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("环境变量 HF_TOKEN 未设置，请设置后重试。")
    login(token=hf_token)
    print("✅ 已成功登录 Hugging Face Hub。")

def main():
    project_root = find_project_root()
    dataset_dir = project_root / "datasets"
    print(f"✅ 项目根目录: {project_root}")
    print(f"✅ 数据集目录: {dataset_dir}")
    check_login_huggingface()
    download_wikipedia_dataset(dataset_dir)
    print("✅ 数据集下载完成！")


if __name__ == "__main__":
    main()