# ============================================================================
# 导入必要的库
# ============================================================================
import torch
import einops  # 用于张量操作的易读性，提供 einsum 等函数
import unittest  # Python 标准单元测试框架
import itertools  # 用于生成笛卡尔积组合
import rapid_attention as ra  # 自定义的 Flash Attention CUDA 算子库
from dataclasses import dataclass  # 用于定义数据类，简化配置对象的创建
from parameterized import parameterized  # 用于参数化测试，让同一个测试函数运行多组参数
from click.core import batch  # 未使用，可能是一个遗留导入
from enum import IntEnum  # 用于定义整数枚举类型


# ============================================================================
# 全局配置常量
# ============================================================================
# 不同序列长度对应的批次大小映射表
# 序列越长，显存占用越大，因此批次大小相应减小
BATCH_SIZE_FOR_SEQ_LEN = {
    512: 16,
    1024: 16,
    2048: 16,
    4096: 16,
    8192: 8,
    16384: 4,
}
# 标准测试中使用的注意力头数
BENCHMARK_N_HEADS = 16


# ============================================================================
# 数据类型枚举类
# ============================================================================
# 这是一个 hack，目的是避免在 C++ 代码中直接依赖 torch
class DType(IntEnum):
    """
    数据类型枚举，映射 PyTorch 的数据类型到整数值
    
    设计目的：
    1. 使数据类型可以在 C++ 和 Python 之间无缝传递
    2. 避免在 C++ 代码中引入 torch 的头文件依赖
    3. 提供统一的类型转换接口
    
    PyTorch ScalarType 枚举参考：
    https://github.com/pytorch/pytorch/blob/c37ddcaefbe9b877e1816ce97dedb8ad26d09450/c10/core/ScalarType.h
    """
    FP16 = 5   # torch.float16 在 PyTorch 内部枚举值为 5
    BF16 = 15  # torch.bfloat16 在 PyTorch 内部枚举值为 15

    def to_cpp_str(self) -> str:
        """
        将 DType 转换为 C++ 代码中使用的字符串表示
        
        用途：生成 C++ 核函数配置结构体时使用
        例如：DType.FP16.to_cpp_str() -> "torch::kFloat16"
        
        Returns:
            C++ 代码中对应的类型字符串
        """
        if self == DType.FP16:
            return "torch::kFloat16"
        elif self == DType.BF16:
            return "torch::kBFloat16"
        else:
            raise ValueError(f"Invalid DType: {self}")

    def to_torch_dtype(self):
        """
        将 DType 转换为 PyTorch 的 torch.dtype 对象
        
        用途：在 Python 端创建张量时使用
        注意：延迟导入 torch，避免在 C++ 端不必要的依赖
        
        Returns:
            torch.dtype: 对应的 PyTorch 数据类型
        """
        import torch

        if self == DType.FP16:
            return torch.float16
        elif self == DType.BF16:
            return torch.bfloat16
        else:
            raise ValueError(f"Invalid DType: {self}")

    @classmethod
    def from_string(cls, dtype_str: str) -> "DType":
        """
        从字符串解析 DType，支持大小写不敏感
        
        支持两种格式：
        1. 名称格式："FP16"、"BF16"（不区分大小写）
        2. 整数格式："5"、"15"（对应枚举值）
        
        Args:
            dtype_str: 数据类型的字符串表示
        
        Returns:
            DType: 解析后的枚举值
        
        Raises:
            ValueError: 如果无法解析输入的字符串
        """
        dtype_str = dtype_str.strip()

        # 首先尝试解析为整数（支持整数格式）
        try:
            dtype_int = int(dtype_str)
            return cls(dtype_int)
        except ValueError:
            pass

        # 然后尝试解析为枚举名称（支持名称格式）
        dtype_str = dtype_str.upper()
        try:
            return cls[dtype_str]
        except KeyError:
            # 如果都失败，给出清晰的错误提示
            valid_options = [
                f"{member.name} ({member.value})" for member in cls
            ]
            raise ValueError(
                f"Invalid dtype string '{dtype_str}'. Valid options: {valid_options}"
            )


# 每个元素占用的字节数（FP16/BF16 均为 2 字节）
ELEM_SIZE = 2  # bytes


# ============================================================================
# QKV 配置数据类
# ============================================================================
@dataclass(frozen=True)
class QKVConfig:
    """
    Query-Key-Value 配置类，用于描述 Attention 计算的所有维度参数
    
    使用 frozen=True 确保配置对象不可变，便于在多线程/多进程环境中安全使用
    """
    n_heads: int          # 注意力头的数量
    d_head: int           # 每个注意力头的维度
    
    batch_size: int       # 批次大小
    seq_len: int          # 序列长度
    
    dtype: torch.dtype    # 数据类型
    device: torch.device  # 计算设备（CPU/CUDA）


# ============================================================================
# 测试数据生成函数
# ============================================================================
def generate_qkv(cfg: QKVConfig):
    """
    生成随机的 Q、K、V 张量用于测试
    
    数据形状：
    - q: (batch_size, seq_len, n_heads, d_head)
    - k: (batch_size, seq_len, n_heads, d_head)
    - v: (batch_size, seq_len, n_heads, d_head)
    
    所有张量使用标准正态分布（均值为 0，方差为 1）初始化
    
    Args:
        cfg: QKVConfig 配置对象
        
    Returns:
        tuple: (q, k, v) 三个随机张量
    """
    q = torch.randn(
        (cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.d_head),
        dtype=cfg.dtype,
        device=cfg.device,
    )
    k = torch.randn_like(q)  # 复用 q 的 shape、dtype、device
    v = torch.randn_like(q)

    return q, k, v


# ============================================================================
# FLOPs 计算函数
# ============================================================================
def calc_total_flop(n_samples, n_heads, seq_len, B_r, B_c, d_head):
    """
    计算 Flash Attention 内核的总浮点运算次数（FLOPs）
    
    这个函数基于分块策略计算理论浮点运算量，用于性能分析和 roofline 模型
    
    算法分解：
    1. Flash Attention 将序列切分为大小为 B_r 和 B_c 的块
    2. T_r = seq_len / B_r：行块数量
    3. T_c = seq_len / B_c：列块数量
    4. 每个 KV 块需要计算 Q 块和 KV 块之间的注意力
    
    复杂度分析：
    - 每个 KV tile 的 FLOPs: kv_tile_flop(B_r, B_c, d_head)
    - 每个 Q 块需要处理所有 T_c 个 KV 块
    - 还需要 epilogue 操作（输出投影）
    
    Args:
        n_samples: 样本数（批次大小）
        n_heads: 注意力头数
        seq_len: 序列长度
        B_r: Q 的分块大小
        B_c: K/V 的分块大小
        d_head: 每个头的维度
    
    Returns:
        int: 总浮点运算次数
    """
    assert seq_len % B_r == 0
    assert seq_len % B_c == 0

    T_r = seq_len // B_r  # 行块数量
    T_c = seq_len // B_c  # 列块数量

    # 每个头、每个样本的 epilogue 操作 FLOPs
    epilogue_flops = B_r * d_head
    
    # 每个头、每个样本的 FLOPs
    # T_r 个行块，每个行块需要处理 T_c 个 KV 块
    head_sample_flops = T_r * (
        T_c * kv_tile_flop(B_r, B_c, d_head) + epilogue_flops
    )

    # 总 FLOPs = 每个头每个样本的 FLOPs * 头数 * 样本数
    return head_sample_flops * n_samples * n_heads


def calc_self_attn_flop(n_samples, n_heads, seq_len, d_head):
    """
    计算标准 Self-Attention 的理论 FLOPs（非 Flash Attention）
    
    标准 Attention 的复杂度：
    1. Q @ K^T: seq_len * seq_len * d_head * 2 (乘加)
    2. Softmax: seq_len * seq_len (约)
    3. P @ V: seq_len * seq_len * d_head * 2 (乘加)
    
    简化的计算公式：
    - 4 * seq_len^2 * d_head + 6 * seq_len^2
    其中 4 来自 Q*K 和 P*V 的乘加操作，6 来自 Softmax 的相关操作
    
    这个函数主要用于与 Flash Attention 的 FLOPs 进行对比，
    衡量 Flash Attention 的算法优化效果
    
    Args:
        n_samples: 样本数
        n_heads: 注意力头数
        seq_len: 序列长度
        d_head: 每个头的维度
    
    Returns:
        int: 标准 Attention 的总浮点运算次数
    """
    return n_samples * n_heads * (4 * seq_len**2 * d_head + 6 * seq_len**2)


# ============================================================================
# Flash Attention 内核配置类（核心数据结构）
# ============================================================================
@dataclass(frozen=True, order=True)
class FlashForwardKernelConfig:
    """
    Flash Attention 前向传播内核的完整配置参数
    
    这个数据类描述了 CUDA 内核的所有可调参数，用于：
    1. 自动调优（Auto-Tuning）：搜索最优配置
    2. 正确性测试：验证不同配置下的计算结果
    3. 性能分析：关联 FLOPs 和实际运行时间
    
    参数详解：
    - dtype: 数据类型（FP16/BF16）
    - d_head: 每个头的维度
    - B_r: Q 的分块大小（行块）
    - B_c: K/V 的分块大小（列块）
    - n_warps: CUDA block 中的 warp 数量（决定线程数）
    - async_copy: 是否使用异步拷贝（掩盖内存延迟）
    - eager_load_blocks: 是否提前加载块（预取策略）
    - swizzled: 是否使用 swizzled 内存访问模式（减少 bank conflict）
    - Q_mma_load_K_tiles: Q 的 MMA 加载 tiles 数量
    - K_mma_load_K_tiles: K 的 MMA 加载 tiles 数量
    - V_mma_load_V_tiles: V 的 MMA 加载 tiles 数量
    - mma_double_buffer_loads: 是否使用 MMA 双缓冲
    - optimized_softmax: 是否使用优化的 Softmax 实现
    
    注意：Q_mma_load_*_tiles 参数控制 MMA（矩阵乘加）指令的
    加载策略，影响寄存器的使用和指令级并行度
    """
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
        """字符串表示，用于测试用例的名称"""
        return self.short_form()

    def short_form(self, include_d_head=True, include_tup=True):
        """
        生成配置的简洁字符串表示
        
        格式示例：
        (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        
        Args:
            include_d_head: 是否包含 d_head 信息
            include_tup: 是否包含元组前缀
        
        Returns:
            str: 配置的字符串描述
        """
        d_head_str = f"{self.d_head}, " if include_d_head else ""
        base = f"({self.dtype.name}, {d_head_str}{self.B_r}, {self.B_c}, {self.n_warps}): "
        if not include_tup:
            base = ""
        
        # 收集所有开启的优化标志
        strs = []
        if self.async_copy:
            strs.append("async")
        if self.eager_load_blocks:
            strs.append("eager")
        if self.swizzled:
            strs.append("swizzled")

        # MMA 加载策略
        strs.append(
            f"load_{self.Q_mma_load_K_tiles}_{self.K_mma_load_K_tiles}_{self.V_mma_load_K_tiles}_tiles"
        )
        if self.mma_double_buffer_loads:
            strs.append("buffer")
        if self.optimized_softmax:
            strs.append("opt_softmax")

        return base + "+".join(strs)

    def to_cpp_struct(self) -> str:
        """
        生成 C++ 结构体初始化代码
        
        这个函数生成的字符串可以直接嵌入到 C++ 代码中，
        用于实例化对应的 CUDA 内核
        
        生成的代码示例：
        FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, 
                                 true, true, true, 2, 2, 2, true, true}
        
        Returns:
            str: C++ 结构体初始化代码
        """
        def vstr(v):
            # 将 Python 值转换为 C++ 字面量
            if isinstance(v, bool):
                return str(v).lower()  # True -> true, False -> false
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
        """返回 CUDA 内核的名称（用于编译和调用）"""
        return "flash_forward_kernel"

    def total_flop(self, n_samples: int, n_heads: int, seq_len: int) -> int:
        """计算当前配置下的总 FLOPs"""
        return calc_total_flop(
            n_samples, n_heads, seq_len, self.B_r, self.B_c, self.d_head
        )

    def attn_flop(self, n_samples: int, n_heads: int, seq_len: int) -> int:
        """计算标准 Attention 的 FLOPs（作为对比基准）"""
        return calc_self_attn_flop(n_samples, n_heads, seq_len, self.d_head)


# ============================================================================
# 数据生成工具函数
# ============================================================================
def generate_qkvo(cfg: QKVConfig):
    """
    生成 Q、K、V、O 四个张量，其中 O 是输出缓冲区（未初始化）
    
    与 generate_qkv 的区别：
    - generate_qkv: 只生成 Q、K、V（输入）
    - generate_qkvo: 额外生成 O（输出），用于就地计算
    
    使用场景：需要预先分配输出张量的性能测试
    
    Args:
        cfg: QKVConfig 配置对象
    
    Returns:
        tuple: (q, k, v, o) 其中 o 是未初始化的输出缓冲区
    """
    # 分配 4 个张量的连续内存块，提高内存访问效率
    all = torch.empty(
        (4, cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.d_head),
        dtype=cfg.dtype,
        device=cfg.device,
    )
    # 拆分出独立的张量视图
    q, o, k, v = tuple(all[i] for i in range(all.size(0)))
    
    # 为 Q、K、V 填充随机值（O 保持未初始化）
    torch.randn(q.shape, dtype=cfg.dtype, device=cfg.device, out=q)
    torch.randn(k.shape, dtype=cfg.dtype, device=cfg.device, out=k)
    torch.randn(v.shape, dtype=cfg.dtype, device=cfg.device, out=v)
    
    return q, k, v, o


# ============================================================================
# PyTorch 参考实现（Ground Truth）
# ============================================================================
def py_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    upcast: bool = False,
):
    """
    PyTorch 标准 Attention 实现，作为测试的参考基准（Ground Truth）
    
    这个实现是"教科书级别"的 Attention，虽然慢但数学上绝对正确。
    用于验证自定义 CUDA 算子的正确性。
    
    计算流程：
    1. S = Q @ K^T / sqrt(d_head)  (计算注意力分数)
    2. P = softmax(S)              (归一化得到注意力权重)
    3. O = P @ V                   (加权求和得到输出)
    
    Args:
        q: Query 张量，形状 (batch, seq_q, head, d_head)
        k: Key 张量，形状 (batch, seq_k, head, d_head)
        v: Value 张量，形状 (batch, seq_k, head, d_head)
        upcast: 是否提升精度到 FP32 计算（提供更高精度的参考）
    
    Returns:
        torch.Tensor: Attention 输出，形状 (batch, seq_q, head, d_head)
    """
    d_head = q.shape[-1]
    dtype_og = q.dtype
    
    # 可选：提升精度，获得更精确的参考结果
    if upcast:
        q, k, v = q.float(), k.float(), v.float()

    # 步骤 1：计算 Q @ K^T，并缩放
    # einops 提供了更易读的 einsum 语法
    S = einops.einsum(
        q,
        k,
        "batch seq_q head d_head, batch seq_k head d_head -> batch seq_q head seq_k",
    ) / (d_head**0.5)
    
    # 步骤 2：Softmax 归一化
    attn_probs = S.softmax(dim=-1)

    # 步骤 3：计算 P @ V
    out = einops.einsum(
        attn_probs,
        v,
        "batch seq_q head seq_k, batch seq_k head d_head -> batch seq_q head d_head",
    )
    
    # 如果之前提升了精度，现在转换回原始数据类型
    if upcast:
        out = out.to(dtype=dtype_og)
    
    return out


# ============================================================================
# 自动调优配置筛选器
# ============================================================================
def should_autotune_config(cfg: FlashForwardKernelConfig) -> bool:
    """
    判断某个内核配置是否应该被纳入自动调优搜索空间
    
    这个函数通过静态规则过滤掉无效或次优的配置，减少搜索空间
    
    过滤规则：
    1. 如果不使用异步拷贝，但使用 eager 加载，配置无效
    2. Q 的加载 tiles 数必须与 K 相同（或 Q 为 0），否则会导致负载不均衡
    3. 针对 B_r=64 的特定规则：
       - n_warps=8 时寄存器压力过大
       - B_c=32 且 Q_mma_load=0 时寄存器超过阈值
       - B_c=64 且 Q_mma_load != 0 时寄存器不足
    4. 针对 B_r=128 的规则：
       - 必须使用 Q_mma_load > 0，否则性能不佳
    
    Args:
        cfg: 待检查的内核配置
    
    Returns:
        bool: True 表示应该测试此配置，False 表示跳过
    """
    # 规则 1：异步拷贝和 eager 加载的兼容性
    if not cfg.async_copy and cfg.eager_load_blocks:
        return False
    
    # 规则 2：Q 和 K 的加载策略必须匹配（负载均衡）
    if (
        cfg.Q_mma_load_K_tiles != cfg.K_mma_load_K_tiles
        and cfg.Q_mma_load_K_tiles != 0
    ):
        return False

    # 规则 3：针对 B_r=64 的特殊优化规则
    if cfg.B_r == 64:
        # n_warps=8 导致寄存器压力过大
        if cfg.n_warps == 8:
            return False
        # B_c=32 且没有 Q_mma_load 会导致寄存器超过阈值
        elif (
            cfg.B_c == 32 and cfg.Q_mma_load_K_tiles == 0
        ):  # over threshold of # registers for 3 CTA
            return False
        # B_c=64 且有 Q_mma_load 会导致寄存器不足
        elif cfg.B_c == 64 and cfg.Q_mma_load_K_tiles != 0:
            return False
    # 规则 4：针对 B_r=128 的特殊优化规则
    elif cfg.B_r == 128:
        # B_r=128 时必须有 Q_mma_load
        if cfg.Q_mma_load_K_tiles == 0:
            return False

    return True


# ============================================================================
# 自动调优配置生成器
# ============================================================================
def get_autotuning_kernel_configs(dtypes=[DType.BF16, DType.FP16]):
    """
    生成所有需要进行自动调优的内核配置
    
    设计思路：
    1. 定义所有可调参数的取值范围
    2. 使用笛卡尔积生成所有组合
    3. 通过过滤器移除无效配置
    
    注意：当前注释掉的参数展示了完整的搜索空间，
    实际使用的参数是经过预筛选的缩小版，用于快速测试
    
    完整的搜索空间（注释中）：
    - d_heads: [128]
    - B_rs: [64, 128]
    - B_cs: [32, 64]
    - n_warps: [4]
    - async_copy: [True]
    - eager_load_blocks: [True]
    - swizzleds: [True]
    - Q_mma_load: [0, 2]
    - K_mma_load: [0, 2]
    - V_mma_load: [0, 2]
    - mma_double_buffer: [False, True]
    - optimized_softmax: [False, True]
    
    实际使用的参数（缩小版，快速测试）：
    - 只测试 B_r=64, B_c=32
    - 固定所有优化标志为 True
    - 固定 MMA 加载策略为 2-2-2
    
    Args:
        dtypes: 要测试的数据类型列表
    
    Returns:
        list[FlashForwardKernelConfig]: 所有有效配置的列表
    """
    # ===== 完整的搜索空间（注释状态） =====
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

    # ===== 实际使用的配置空间（快速测试版本） =====
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

    # 所有参数列表（顺序与 FlashForwardKernelConfig 的字段顺序一致）
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

    # 生成所有组合并应用过滤器
    return [
        FlashForwardKernelConfig(*cfg)
        for cfg in itertools.product(*params)
        if should_autotune_config(FlashForwardKernelConfig(*cfg))
    ]


def get_kernels_to_build():
    """
    获取所有需要编译的内核配置列表
    
    这里可以组合多个配置来源：
    1. 自动调优配置（当前使用）
    2. 渐进式配置（注释状态，可能用于更系统的测试）
    
    Returns:
        list[FlashForwardKernelConfig]: 所有需要构建的配置（已排序）
    """
    cfgs = set()
    # cfgs.update(get_kernel_progression_configs())  # 可选的额外配置来源
    cfgs.update(get_autotuning_kernel_configs())

    return sorted(cfgs)


# ============================================================================
# 测试基类
# ============================================================================
class BaseRapidAttentionTest:
    """
    Flash Attention 测试的基类，提供通用的测试逻辑
    
    设计模式：模板方法模式（Template Method Pattern）
    - 基类定义测试流程（setUpClass 和 _test_standard）
    - 子类实现具体的数据类型（dtype 方法）
    
    测试策略：差分测试（Differential Testing）
    1. 使用 PyTorch 实现作为参考（FP32 和 FP16）
    2. 比较自定义算子与 PyTorch 实现的误差
    3. 误差容忍度 = PyTorch FP16 vs FP32 误差 * 2
    """

    @classmethod
    def setUpClass(cls):
        """
        测试类的初始化方法（在所有测试方法之前执行）
        
        工作流程：
        1. 根据 seq_len 确定合适的 batch_size
        2. 生成随机的 Q、K、V 数据
        3. 计算 PyTorch 的参考结果（FP16 和 FP32）
        
        数据存储为类变量，所有测试用例共享，确保一致性
        """
        seq_len = 2048
        batch_size = BATCH_SIZE_FOR_SEQ_LEN[seq_len]
        n_heads = BENCHMARK_N_HEADS
        dtype = cls.dtype()
        device = "cuda:0"

        cls.d_heads = [128]
        cls.data = {}          # 存储 Q、K、V 数据
        cls.pt_b16_results = {}  # PyTorch BF16/FP16 结果
        cls.pt_f32_results = {}  # PyTorch FP32 结果（高精度参考）
        
        for d_head in cls.d_heads:
            cfg = QKVConfig(
                n_heads=n_heads,
                d_head=d_head,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=dtype,
                device=device
            )

            # 生成测试数据
            q, k, v = generate_qkv(cfg)
            cls.data[d_head] = (q, k, v)
            
            # 计算 PyTorch 参考结果
            cls.pt_b16_results[d_head] = py_flash_attention(q, k, v, upcast=False)
            cls.pt_f32_results[d_head] = py_flash_attention(q, k, v, upcast=True)

    def _test_starndard(self, name, cfg):
        """
        标准的正确性测试方法
        
        测试逻辑：
        1. 使用自定义 CUDA 算子计算 Flash Attention
        2. 计算与 PyTorch FP16 结果的差值（diff_fb16）
        3. 计算 PyTorch FP16 与 FP32 的差值（diff_fb32）
        4. 断言：diff_fb16 <= diff_fb32 * 2
        
        这个测试的核心原理：
        - 如果自定义算子的误差不超过 PyTorch 自身的精度误差的 2 倍，
          则认为结果是正确的
        
        Args:
            name: 测试用例名称（用于显示）
            cfg: FlashForwardKernelConfig 配置
        """
        # 获取测试数据
        q, k, v = self.__class__.data[cfg.d_head]
        
        # 运行自定义 CUDA 算子
        result = ra.forward(cfg, q, k, v)
        
        # 获取 PyTorch 参考结果
        fp16_result = self.__class__.pt_b16_results[cfg.d_head]
        fp32_result = self.__class__.pt_f32_results[cfg.d_head]
        
        # 计算误差（使用无穷范数）
        diff_fb16 = (result - fp16_result).abs().max().item()
        diff_fb32 = (fp16_result - fp32_result).abs().max().item()

        # 断言：自定义算子的误差不应超过 PyTorch 精度误差的 2 倍
        self.assertLessEqual(diff_fb16, diff_fb32 * 2)

    @classmethod
    def dtype(self):
        """
        抽象方法：子类必须实现，返回要测试的数据类型
        
        子类应该返回 torch.float16 或 torch.bfloat16
        """
        raise NotImplementedError("请在子类中实现 dtype 方法，返回 torch.float16 或 torch.bfloat16")


# ============================================================================
# 具体的测试用例类
# ============================================================================
class RapidAttentionFP16Test(BaseRapidAttentionTest, unittest.TestCase):
    """
    FP16 数据类型的 Flash Attention 测试类
    
    继承自：
    - BaseRapidAttentionTest: 提供通用测试逻辑
    - unittest.TestCase: 提供 unittest 框架支持
    """

    @classmethod
    def dtype(cls):
        """返回 FP16 数据类型"""
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
        """
        参数化测试：自动为每个配置生成独立的测试用例
        
        parameterized.expand 的作用：
        - 将列表中的每个元素展开为一个独立的测试
        - 测试名称会自动包含配置信息
        - skip_on_empty=True：如果没有配置，自动跳过测试
        
        这使得我们可以用一个测试函数覆盖所有内核配置，
        每个配置都能独立报告成功/失败
        """
        self._test_starndard(name, cfg)


"""
# 注释掉的 BF16 测试类
# 取消注释后可以同时测试 BF16 数据类型
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
"""


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == "__main__":
    """
    当脚本直接运行时，执行单元测试
    
    verbosity=2：详细输出模式，显示每个测试用例的名称和结果
    """
    unittest.main(verbosity=2)