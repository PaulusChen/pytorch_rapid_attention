import os
import torch
from pathlib import Path


def set_pytorch_ld_library_path():
    """设置PyTorch的LD_LIBRARY_PATH"""
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path):
        current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        paths = current_ld_library_path.split(os.pathsep) if current_ld_library_path else []
        if torch_lib_path not in paths:
            paths.insert(0, torch_lib_path)
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(paths)


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

