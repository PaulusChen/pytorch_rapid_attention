
import torch
import einops
import unittest
import itertools
import rapid_attention as ra
from dataclasses import dataclass
from parameterized import parameterized
from click.core import batch
from enum import IntEnum


BATCH_SIZE_FOR_SEQ_LEN = {
    512: 16,
    1024: 16,
    2048: 16,
    4096: 16,
    8192: 8,
    16384: 4,
}
BENCHMARK_N_HEADS = 16


# This is a hack to avoid importing torch.
class DType(IntEnum):
    # https://github.com/pytorch/pytorch/blob/c37ddcaefbe9b877e1816ce97dedb8ad26d09450/c10/core/ScalarType.h
    # These are the enum values for the torch types
    FP16 = 5
    BF16 = 15

    def to_cpp_str(self) -> str:
        if self == DType.FP16:
            return "torch::kFloat16"
        elif self == DType.BF16:
            return "torch::kBFloat16"
        else:
            raise ValueError(f"Invalid DType: {self}")

    def to_torch_dtype(self):
        import torch

        if self == DType.FP16:
            return torch.float16
        elif self == DType.BF16:
            return torch.bfloat16
        else:
            raise ValueError(f"Invalid DType: {self}")

    @classmethod
    def from_string(cls, dtype_str: str) -> "DType":
        """Parse DType from string. Case-insensitive. Handles both names ('FP16', 'BF16') and integers ('0', '1')."""
        dtype_str = dtype_str.strip()

        # Try parsing as integer first
        try:
            dtype_int = int(dtype_str)
            return cls(dtype_int)
        except ValueError:
            pass

        # Try parsing as enum name
        dtype_str = dtype_str.upper()
        try:
            return cls[dtype_str]
        except KeyError:
            valid_options = [
                f"{member.name} ({member.value})" for member in cls
            ]
            raise ValueError(
                f"Invalid dtype string '{dtype_str}'. Valid options: {valid_options}"
            )


ELEM_SIZE = 2  # bytes


@dataclass(frozen=True)
class QKVConfig:
    n_heads: int
    d_head: int

    batch_size: int
    seq_len: int

    dtype: torch.dtype
    device: torch.device


def generate_qkv(cfg: QKVConfig):
    q = torch.randn(
        (cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.d_head),
        dtype=cfg.dtype,
        device=cfg.device,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    return q, k, v


def calc_total_flop(n_samples, n_heads, seq_len, B_r, B_c, d_head):
    assert seq_len % B_r == 0
    assert seq_len % B_c == 0

    T_r = seq_len // B_r
    T_c = seq_len // B_c

    epilogue_flops = B_r * d_head
    head_sample_flops = T_r * (
        T_c * kv_tile_flop(B_r, B_c, d_head) + epilogue_flops
    )

    return head_sample_flops * n_samples * n_heads


def calc_self_attn_flop(n_samples, n_heads, seq_len, d_head):
    return n_samples * n_heads * (4 * seq_len**2 * d_head + 6 * seq_len**2)


@dataclass(frozen=True, order=True)
class FlashForwardKernelConfig:
    dtype: DType
    d_head: int
    B_r: int
    B_c: int
    n_warps: int
    async_copy: bool
    eager_load_blocks: bool
    swizzled: bool
    Q_mma_load_K_tiles: int
    K_mma_load_K_tiles: int
    V_mma_load_K_tiles: int
    mma_double_buffer_loads: bool
    optimized_softmax: bool

    def __str__(self):
        return self.short_form()

    def short_form(self, include_d_head=True, include_tup=True):
        d_head_str = f"{self.d_head}, " if include_d_head else ""
        base = f"({self.dtype.name}, {d_head_str}{self.B_r}, {self.B_c}, {self.n_warps}): "
        if not include_tup:
            base = ""
        strs = []
        if self.async_copy:
            strs.append("async")
        if self.eager_load_blocks:
            strs.append("eager")
        if self.swizzled:
            strs.append("swizzled")

        strs.append(
            f"load_{self.Q_mma_load_K_tiles}_{self.K_mma_load_K_tiles}_{self.V_mma_load_K_tiles}_tiles"
        )
        if self.mma_double_buffer_loads:
            strs.append("buffer")
        if self.optimized_softmax:
            strs.append("opt_softmax")

        return base + "+".join(strs)

    def to_cpp_struct(self) -> str:
        def vstr(v):
            if isinstance(v, bool):
                return str(v).lower()
            else:
                return str(v)

        return (
            f"FlashForwardKernelConfig{{"
            f"{self.dtype.to_cpp_str()}, {self.d_head}, {self.B_r}, {self.B_c}, {self.n_warps}, "
            f"{vstr(self.async_copy)}, {vstr(self.eager_load_blocks)}, "
            f"{vstr(self.swizzled)}, {self.Q_mma_load_K_tiles}, {self.K_mma_load_K_tiles}, "
            f"{self.V_mma_load_K_tiles}, {vstr(self.mma_double_buffer_loads)}, "
            f"{vstr(self.optimized_softmax)}"
            f"}}"
        )

    def kernel_name(self) -> str:
        return "flash_forward_kernel"

    def total_flop(self, n_samples: int, n_heads: int, seq_len: int) -> int:
        return calc_total_flop(
            n_samples, n_heads, seq_len, self.B_r, self.B_c, self.d_head
        )

    def attn_flop(self, n_samples: int, n_heads: int, seq_len: int) -> int:
        return calc_self_attn_flop(n_samples, n_heads, seq_len, self.d_head)


def generate_qkvo(cfg: QKVConfig):
    all = torch.empty(
        (4, cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.d_head),
        dtype=cfg.dtype,
        device=cfg.device,
    )
    q, o, k, v = tuple(all[i] for i in range(all.size(0)))
    torch.randn(q.shape, dtype=cfg.dtype, device=cfg.device, out=q)
    torch.randn(k.shape, dtype=cfg.dtype, device=cfg.device, out=k)
    torch.randn(v.shape, dtype=cfg.dtype, device=cfg.device, out=v)
    return q, k, v, o


def py_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    upcast: bool = False,
):
    d_head = q.shape[-1]
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()

    S = einops.einsum(
        q,
        k,
        "batch seq_q head d_head, batch seq_k head d_head -> batch seq_q head seq_k",
    ) / (d_head**0.5)
    attn_probs = S.softmax(dim=-1)

    out = einops.einsum(
        attn_probs,
        v,
        "batch seq_q head seq_k, batch seq_k head d_head -> batch seq_q head d_head",
    )
    if upcast:
        out = out.to(dtype=dtype_og)
    return out


def should_autotune_config(cfg: FlashForwardKernelConfig) -> bool:
    if not cfg.async_copy and cfg.eager_load_blocks:
        return False
    if (
        cfg.Q_mma_load_K_tiles != cfg.K_mma_load_K_tiles
        and cfg.Q_mma_load_K_tiles != 0
    ):
        return False

    if cfg.B_r == 64:
        if cfg.n_warps == 8:
            return False
        elif (
            cfg.B_c == 32 and cfg.Q_mma_load_K_tiles == 0
        ):  # over threshold of # registers for 3 CTA
            return False
        elif cfg.B_c == 64 and cfg.Q_mma_load_K_tiles != 0:
            return False
    elif cfg.B_r == 128:
        if cfg.Q_mma_load_K_tiles == 0:
            return False

    return True


def get_autotuning_kernel_configs(dtypes=[DType.BF16, DType.FP16]):

    """
    d_heads = [128]
    B_rs = [64, 128]
    B_cs = [32, 64]
    n_warps_cfgs = [4]
    async_copy = [True]
    eager_load_blocks = [True]
    swizzleds = [True]
    Q_mma_load_K_tiles = [0, 2]
    K_mma_load_K_tiles = [0, 2]
    V_mma_load_K_tiles = [0, 2]
    mma_double_buffer_loads = [False, True]
    optimized_softmax = [False, True]
    """

    d_heads = [128]
    B_rs = [64]
    B_cs = [32]
    n_warps_cfgs = [4]
    async_copy = [True]
    eager_load_blocks = [True]
    swizzleds = [True]
    Q_mma_load_K_tiles = [2]
    K_mma_load_K_tiles = [2]
    V_mma_load_K_tiles = [2]
    mma_double_buffer_loads = [True]
    optimized_softmax = [True]

    params = [
        dtypes,
        d_heads,
        B_rs,
        B_cs,
        n_warps_cfgs,
        async_copy,
        eager_load_blocks,
        swizzleds,
        Q_mma_load_K_tiles,
        K_mma_load_K_tiles,
        V_mma_load_K_tiles,
        mma_double_buffer_loads,
        optimized_softmax,
    ]

    return [
        FlashForwardKernelConfig(*cfg)
        for cfg in itertools.product(*params)
        if should_autotune_config(FlashForwardKernelConfig(*cfg))
    ]


def get_kernels_to_build():
    cfgs = set()
    # cfgs.update(get_kernel_progression_configs())
    cfgs.update(get_autotuning_kernel_configs())

    return sorted(cfgs)


class BaseRapidAttentionTest:
    @classmethod
    def setUpClass(cls):
        seq_len = 2048
        batch_size = BATCH_SIZE_FOR_SEQ_LEN[seq_len]
        n_heads = BENCHMARK_N_HEADS
        dtype = cls.dtype()
        device = "cuda:0"

        cls.d_heads = [128]
        cls.data = {}
        cls.pt_b16_results = {}
        cls.pt_f32_results = {}
        for d_head in cls.d_heads:
            cfg = QKVConfig(
                n_heads=n_heads,
                d_head=d_head,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=dtype,
                device=device
            )

            q, k, v = generate_qkv(cfg)
            cls.data[d_head] = (q, k, v)
            cls.pt_b16_results[d_head] = py_flash_attention(q, k, v, upcast=False)
            cls.pt_f32_results[d_head] = py_flash_attention(q, k, v, upcast=True)


    def _test_starndard(self, name, cfg):
        q, k, v = self.__class__.data[cfg.d_head]
        result = ra.flash_attention(cfg, q, k, v)
        fp16_result = self.__class__.pt_b16_results[cfg.d_head]
        fp32_result = self.__class__.pt_f32_results[cfg.d_head]
        
        diff_fb16 = (result - fp16_result).abs().max().item()
        diff_fb32 = (result - fp32_result).abs().max().item()

        self.assertLessEqual(diff_fb16, diff_fb32 * 2)


    @classmethod
    def dtype(self):
        raise NotImplementedError("请在子类中实现 dtype 方法，返回 torch.float16 或 torch.bfloat16")


class RapidAttentionFP16Test(BaseRapidAttentionTest, unittest.TestCase):
    @classmethod
    def dtype(cls):
        return torch.float16

    @parameterized.expand(
        [
            (str(cfg), cfg)
            for cfg in get_kernels_to_build()
            if cfg.dtype == DType.FP16
        ],
        skip_on_empty=True,
    )
    def test_fb16(self, name, cfg):
        self._test_starndard(name, cfg)


class RapidAttentionBF16Test(BaseRapidAttentionTest, unittest.TestCase):
    @classmethod
    def dtype(cls):
        return torch.bfloat16

    @parameterized.expand(
        [
            (str(cfg), cfg)
            for cfg in get_kernels_to_build()
            if cfg.dtype == DType.BF16
        ],
        skip_on_empty=True,
    )
    def test_bf16(self, name, cfg):
        self._test_starndard(name, cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)