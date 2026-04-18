from torch import save
import os
import modelscope.hub.snapshot_download as MSDownload
from pathlib import Path
import huggingface_hub as HFH
import rapid_attention as RA


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def check_login_huggingface():
    from huggingface_hub import login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("环境变量 HF_TOKEN 未设置，请设置后重试。")
    login(token=hf_token)
    print("✅ 已成功登录 Hugging Face Hub。")

def download_wikipedia_dataset(datasets_dir: Path):
    datasets_dir.mkdir(exist_ok=True)
    local_path = datasets_dir / "zh_wiki"
    if local_path.exists():
        print("✅ 中文维基数据集已存在，跳过下载。")
        return
    print("📥 开始下载中文维基数据集...")
    try:
        check_login_huggingface()
        download_path = HFH.snapshot_download(
            repo_id="wikimedia/wikipedia",  # 数据集仓库ID
            repo_type="dataset",  # 明确是数据集（不是模型）
            revision="main",  # 分支
            allow_patterns="20231101.zh/",  # 只下载中文维基目录，避免下其他语言
            local_dir=local_path,  # 本地保存目录
        )

    except Exception as e:
        print(f"❌ 下载出错：{str(e)}")


def download_deepctrl_sft_dataset(datasets_dir: Path):
    datasets_dir.mkdir(exist_ok=True)
    dataset_name = "deepctrl/deepctrl-sft-data"
    save_dir = datasets_dir / dataset_name
    if save_dir.exists():
        print(f"✅ 数据集 {dataset_name} 已存在，跳过下载。")
        return
    print(f"📥 开始下载数据集 {dataset_name} ...")
    try:
        # https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py#L155
        download_path = MSDownload.dataset_snapshot_download(
            dataset_id=dataset_name,
            local_dir=save_dir,
        )

    except Exception as e:
        print(f"❌ modelscope 下载数据集 {dataset_name} 出错：{str(e)}")

def download_stepfun_sft_dataset(datasets_dir: Path):
    datasets_dir.mkdir(exist_ok=True)
    dataset_name = "stepfun-ai/Step-3.5-Flash-SFT"
    save_dir = datasets_dir / dataset_name
    if save_dir.exists():
        print(f"✅ 数据集 {dataset_name} 已存在，跳过下载。")
        return
    print(f"📥 开始下载数据集 {dataset_name} ...")
    try:
        # https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py#L155
        download_path = MSDownload.dataset_snapshot_download(
            dataset_id=dataset_name,
            allow_file_pattern="json/general/*",  # 只下载 json/general 目录下的文件
            local_dir=save_dir,
        )

    except Exception as e:
        print(f"❌ modelscope 下载数据集 {dataset_name} 出错：{str(e)}")


def main():
    project_root = RA.utils.common.find_project_root()
    dataset_dir = project_root / "datasets"
    print(f"✅ 项目根目录: {project_root}")
    print(f"✅ 数据集目录: {dataset_dir}")
    download_wikipedia_dataset(dataset_dir)
    download_deepctrl_sft_dataset(dataset_dir)
    download_stepfun_sft_dataset(dataset_dir)
    print("✅ 数据集下载完成！")


if __name__ == "__main__":
    main()