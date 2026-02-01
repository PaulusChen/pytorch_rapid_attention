
import os
import torch
import glob

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "rapid_attention"

py_limited_api = bool(torch.__version__ < "2.6.0")

debug_mode = os.environ.get("DEBUG", "0") == "1"
use_cuda = os.getenv("USE_CUDA", "1") == "1"
if debug_mode:
    print("Building in debug mode")

use_cuda = use_cuda and torch.cuda.is_available() and (CUDA_HOME is not None)
if not use_cuda:
    print("CUDA is not available, quiting build.")
    os.exit(0)

extension_type = CUDAExtension


# define TORCH_TARGET_VERSION with min version 2.10 to expose only the
# stable API subset from torch
# Format: [MAJ 1 byte][MIN 1 byte][PATCH 1 byte][ABI TAG 5 bytes]
# 2.10.0 = 0x020A000000000000
torch_target_version = 0x020a000000000000

extra_link_args = []
extra_compile_args = {
    "cxx": [
        "-O3" if not debug_mode else "-O0",
        "-std=c++20",
        "-fdiagnostics-color=always",
        "-DPy_LIMITED_API=0x03090000",
        f"-DTORCH_TARGET_VERSION=0x{torch_target_version:016x}",
    ], 
    "nvcc": [
        "-O3" if not debug_mode else "-O0",
        "-std=c++20",
        # define TORCH_TARGET_VERSION with min version 2.10 to expose only the
        # stable API subset from torch
        # Format: [MAJ 1 byte][MIN 1 byte][PATCH 1 byte][ABI TAG 5 bytes]
        # 2.10.0 = 0x020A000000000000
        f"-DTORCH_TARGET_VERSION=0x{torch_target_version:016x}",
        "-DUSE_CUDA",
    ]
}

if debug_mode:
    extra_compile_args["cxx"] += ["-g"]
    extra_compile_args["nvcc"] += ["-g"]
    extra_link_args += ["-O0", "-g"]

this_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(this_dir, library_name)

csrc_dir = os.path.join(package_dir, "csrc")
cpp_sources = glob.glob(os.path.join(csrc_dir, "*.cpp"), recursive=True)
print(f"!!!!!!!!!!cpp_sources: {cpp_sources}")

cuda_dir = os.path.join(csrc_dir, "cuda")
cuda_sources = glob.glob(os.path.join(cuda_dir, "*.cu"), recursive=True)
print(f"!!!!!!!!!!cuda_sources: {cuda_sources}")

if use_cuda:
    sources = cpp_sources + cuda_sources

print(f"Compiling {library_name} with CUDA support")

ext_modules = [
    extension_type(
        f"{library_name}._C",
        sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        py_limited_api=py_limited_api,
    )
]

setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=open("requirements.txt").read().splitlines(),
    description="Rapid Attention for PyTorch",
    long_description=open("README.md").read(),
    author="PaulChen",
    author_email="chenpengsmail@qq.com",
    url="https://github.com/PaulusChen/pytorch_rapid_attention",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)

