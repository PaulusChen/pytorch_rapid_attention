from . import _C, ops  # noqa: F401
from .utils.common import set_pytorch_ld_library_path, find_project_root, get_lr
from .utils.logger import logger

set_pytorch_ld_library_path()

def forward(kernel_cfg, q, k, v, o=None):
    return _C.forward(
        kernel_cfg, q, k, v, o, benchmark=False
    )[0]


def forward_timed(kernel_cfg, q, k, v, o=None):
    val, runtime = _C.forward(
        kernel_cfg, q, k, v, o, benchmark=True
    )
    return val, runtime


__all__ = [
    find_project_root,
    logger,
    get_lr,
    forward,
    forward_timed
]

