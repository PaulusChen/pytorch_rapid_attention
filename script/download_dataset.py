from _pytest._py.path import local
from _pytest.cacheprovider import cache
from torch._inductor.lowering import rev
from torch.ao.ns.fx.n_shadows_utils import SHADOW_WRAPPER_NODE_NAME_PREFIX
from torch.__config__ import show
from torch import save
import os
import modelscope.hub.snapshot_download as MSDownload
from pathlib import Path
import huggingface_hub as HFH
import rapid_attention as RA


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download_wikipedia_dataset(datasets_dir: Path):
    datasets_dir.mkdir(exist_ok=True)
    print("📥 开始下载中文维基数据集...")
    try:
        download_path = HFH.snapshot_download(
            repo_id="wikimedia/wikipedia",  # 数据集仓库ID
            repo_type="dataset",  # 明确是数据集（不是模型）
            revision="main",  # 分支
            allow_patterns="20231101.zh/",  # 只下载中文维基目录，避免下其他语言
            local_dir=datasets_dir / "zh_wiki",  # 本地保存目录
        )

    except Exception as e:
        print(f"❌ 下载出错：{str(e)}")


def download_deepctrl_sft_dataset(datasets_dir: Path):
    datasets_dir.mkdir(exist_ok=True)
    dataset_name = "deepctrl/deepctrl-sft-data"
    save_dir = datasets_dir / dataset_name
    print(f"📥 开始下载数据集 {dataset_name} ...")
    try:
        # https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py#L155
        download_path = MSDownload.dataset_snapshot_download(
            dataset_id=dataset_name,
            local_dir=save_dir,
        )

    except Exception as e:
        print(f"❌ modelscope 下载数据集 {dataset_name} 出错：{str(e)}")

def check_login_huggingface():
    from huggingface_hub import login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("环境变量 HF_TOKEN 未设置，请设置后重试。")
    login(token=hf_token)
    print("✅ 已成功登录 Hugging Face Hub。")

def main():
    project_root = RA.utils.common.find_project_root()
    dataset_dir = project_root / "datasets"
    print(f"✅ 项目根目录: {project_root}")
    print(f"✅ 数据集目录: {dataset_dir}")
    check_login_huggingface()
    download_wikipedia_dataset(dataset_dir)
    download_deepctrl_sft_dataset(dataset_dir)
    print("✅ 数据集下载完成！")


if __name__ == "__main__":
    main()