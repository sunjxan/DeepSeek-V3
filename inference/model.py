import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelArgs:
    """
    模型参数配置类，用于定义模型超参数和结构参数

    参数:
        max_batch_size (int): 最大批处理大小，决定缓存分配
        max_seq_len (int): 最大序列长度，用于位置编码和缓存
        dtype (Literal["bf16", "fp8"]): 计算数据类型（当前被注释）
        vocab_size (int): 词表大小，决定嵌入层维度
        dim (int): 模型隐藏层维度
        inter_dim (int): MLP中间层维度
        moe_inter_dim (int): MoE专家网络的中间层维度
        n_layers (int): 模型总层数（Transformer块数量）
        n_dense_layers (int): 使用普通MLP的层数（前n层使用MLP，后续层使用MoE）
        n_heads (int): 注意力头的数量
        # MoE相关参数
        n_routed_experts (int): 路由专家的总数
        n_shared_experts (int): 共享专家数量（所有输入都会经过的专家）
        n_activated_experts (int): 每个token实际使用的专家数量
        n_expert_groups (int): 专家分组数量（用于分布式计算）
        n_limited_groups (int): 每个token可以选择的最大专家组数量
        score_func (Literal["softmax", "sigmoid"]): 路由分数计算方式（当前被注释）
        route_scale (float): 路由权重的缩放因子
        # 注意力相关参数
        q_lora_rank (int): 查询矩阵的LoRA降维秩（0表示不使用LoRA）
        kv_lora_rank (int): 键值矩阵的LoRA降维秩
        qk_nope_head_dim (int): 无位置编码的Q/K头维度
        qk_rope_head_dim (int): 带旋转位置编码的Q/K头维度
        v_head_dim (int): 值向量的头维度
        # YARN扩展相关参数
        original_seq_len (int): 原始训练的序列长度
        rope_theta (float): 旋转位置编码的基础频率
        rope_factor (float): 序列长度扩展因子
        beta_fast (int): 快速beta调整参数
        beta_slow (int): 慢速beta调整参数
        mscale (float): 注意力缩放因子
    """
    max_batch_size: int = 8          # 根据GPU显存调整
    max_seq_len: int = 1024 * 4      # 支持最大4096 tokens
    # dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 10240          # 根据词表实际大小设置
    dim: int = 200                   # 隐藏层维度
    inter_dim: int = 300             # MLP中间层扩展维度
    moe_inter_dim: int = 250         # MoE专家中间层维度
    n_layers: int = 6                # 模型总层数
    n_dense_layers: int = 1          # 前1层使用普通MLP
    n_heads: int = 16                # 注意力头数
    
    # MoE参数配置
    n_routed_experts: int = 64       # 总专家数量
    n_shared_experts: int = 2        # 共享专家数量
    n_activated_experts: int = 6     # 每个token激活的专家数
    n_expert_groups: int = 1         # 专家分组（分布式训练用）
    n_limited_groups: int = 1        # 每个token可选的最大组数
    # score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.          # 路由权重缩放因子
    
    # 注意力机制参数
    q_lora_rank: int = 0             # 查询LoRA秩（0表示禁用）
    kv_lora_rank: int = 512          # 键值LoRA秩
    qk_nope_head_dim: int = 128      # 无位置编码头维度
    qk_rope_head_dim: int = 64       # 旋转位置编码头维度
    v_head_dim: int = 128            # 值向量头维度
    
    # YARN扩展参数（支持长上下文）
    original_seq_len: int = 4096     # 基础训练长度
    rope_theta: float = 10000.0      # RoPE基础频率
    rope_factor: float = 40          # 长度扩展因子
    beta_fast: int = 32              # 快速调整参数
    beta_slow: int = 1               # 慢速调整参数
    mscale: float = 1.               # 注意力缩放因子

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    预计算旋转位置编码的复数形式频率矩阵
    
    算法流程：
    1. 根据基础参数计算初始频率
    2. 如果序列长度超过原始长度，进行长度外推修正
    3. 生成所有位置的频率矩阵
    4. 转换为复数形式（cos + i*sin）
    
    参数:
        args (ModelArgs): 包含所有位置编码参数的模型参数
        
    返回:
        torch.Tensor: 预计算好的复数频率矩阵，形状为(seq_len, dim//2)
    """
    dim = args.qk_rope_head_dim       # 使用带位置编码的头维度
    seqlen = args.max_seq_len         # 最大序列长度
    beta_fast = args.beta_fast        # 长度外推参数
    beta_slow = args.beta_slow
    base = args.rope_theta            # 10000
    factor = args.rope_factor         # 扩展因子

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        计算旋转位置嵌入中给定旋转次数所需的校正维度。（用于长度外推）
        
        公式推导：
        d_c = dim * ln(max_seq_len / (num_rotations * 2π)) / (2 ln(base))
        
        参数:
            num_rotations (float): 需要计算校正的旋转次数
            dim (int): 嵌入空间的维度
            base (float): 指数计算的基础值
            max_seq_len (int): 最大序列长度
        
        返回:
            float: 基于输入参数的校正维度
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        计算旋转位置嵌入的校正维度范围。（生成平滑过渡区域）
        
        参数:
            low_rot (float): 旋转次数的下限
            high_rot (float): 旋转次数的上限
            dim (int): 嵌入空间的维度
            base (float): 指数计算的基础值
            max_seq_len (int): 最大序列长度
        
        返回:
            Tuple[int, int]: 校正维度的范围（下限，上限），已截断到有效索引范围内
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        生成线性斜坡掩码，用于平滑混合基础频率和修正频率
        
        参数:
            min (float): 斜坡函数的最小值
            max (float): 斜坡函数的最大值
            dim (int): 斜坡张量的维度
        
        返回:
            torch.Tensor: 形状为 (dim,) 的张量，值在0到1之间线性插值，并截断到[0,1]范围
        """
        if min == max:  # 避免除零
            max += 0.001
        # 生成0-1的线性插值，然后裁剪到[0,1]区间
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        return torch.clamp(linear_func, 0, 1)

    # 基础频率计算：1/(base^(2i/dim))，i从0到dim//2-1
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    # 长度外推处理（当实际长度超过原始训练长度时）
    if seqlen > args.original_seq_len:
        # 计算需要调整的维度范围
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        # 生成平滑过渡掩码（中间区域混合两种频率）
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        # 混合基础频率和调整后的频率
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 生成位置-频率矩阵
    t = torch.arange(seqlen)  # 所有位置索引
    freqs = torch.outer(t, freqs)  # 外积得到(seqlen, dim//2)矩阵
    # 转换为复数形式：e^(iθ) = cosθ + i sinθ
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量
    
    实现步骤：
    1. 将输入转换为复数形式
    2. 与预计算的频率复数相乘（相当于旋转向量）
    3. 转换回实数形式
    
    参数:
        x (torch.Tensor): 输入张量，形状为(..., head_dim)
        freqs_cis (torch.Tensor): 预计算的复数频率矩阵
    
    返回:
        torch.Tensor: 旋转后的张量，保持原始形状
    """
    dtype = x.dtype  # 保存原始数据类型
    # 将输入转换为复数形式（假设最后两维可以分成实部和虚部）
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # 调整频率矩阵形状以匹配输入（添加广播维度）
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # 复数乘法实现旋转，然后转换回实数
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)  # 恢复原始数据类型

class MLA(nn.Module):
    """
    多头注意力层（Multi-head Latent Attention），包含LoRA支持和KV缓存
    
    结构特点：
    - 可选LoRA低秩适应：当q_lora_rank>0时启用查询LoRA
    - 分离的位置编码处理：将Q分为带位置编码和不带位置编码部分
    - 动态KV缓存管理：缓存历史K/V值用于自回归生成
    
    参数:
        dim (int): 输入特征的维度
        n_heads (int): 注意力头数
        n_local_heads (int): 分布式系统中本地注意力头数
        q_lora_rank (int): 低秩查询投影的秩
        kv_lora_rank (int): 低秩键/值投影的秩
        qk_nope_head_dim (int): 非位置敏感查询/键投影的维度
        qk_rope_head_dim (int): 旋转位置敏感查询/键投影的维度
        qk_head_dim (int): 查询/键投影的总维度
        v_head_dim (int): 值投影的维度
        softmax_scale (float): 注意力计算中softmax的缩放因子
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads        # 总头数
        self.n_local_heads = args.n_heads  # 本地头数（分布式场景使用）
        
        # LoRA配置
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        
        # 头维度分解
        self.qk_nope_head_dim = args.qk_nope_head_dim  # 无位置编码部分
        self.qk_rope_head_dim = args.qk_rope_head_dim  # 带位置编码部分
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim  # Q/K总维度
        self.v_head_dim = args.v_head_dim             # V头维度
        
        # 查询投影（可选LoRA）
        if self.q_lora_rank == 0:  # 标准线性变换
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:  # LoRA分解：W = W_a * W_b
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
            self.q_norm = nn.RMSNorm(self.q_lora_rank)  # 归一化层
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        
        # 键值投影（统一处理）
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank)  # 归一化
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        
        # 输出投影
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        
        # 注意力缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5  # 1/sqrt(d_k)
        # 长上下文缩放调整
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale *= mscale ** 2  # 平方缩放

        # 注册KV缓存（非持久化缓冲区）
        self.register_buffer("k_cache", torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim
        ), persistent=False)
        self.register_buffer("v_cache", torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim
        ), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        前向传播过程
        
        处理步骤：
        1. 投影得到查询Q，分解为无位置编码和有位置编码部分
        2. 投影得到键K和值V，分离位置编码部分
        3. 更新KV缓存
        4. 计算注意力分数
        5. 应用mask（训练时）和softmax
        6. 聚合价值信息并投影输出
        
        参数:
            x: 输入张量 (batch_size, seq_len, dim)
            start_pos: 当前输入的起始位置（用于缓存）
            freqs_cis: 预计算的旋转位置编码
            mask: 注意力mask（防止看到未来信息）
        
        返回:
            输出张量，形状与输入相同
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # 查询投影（带可选LoRA）
        if self.q_lora_rank == 0:
            q = self.wq(x)  # (bsz, seqlen, n_heads*qk_head_dim)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        
        # 分解Q为无位置编码和有位置编码部分
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)  # 应用旋转位置编码
        
        # 键值投影
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # 添加头维度后应用RoPE
        
        # 合并Q的两个部分
        q = torch.cat([q_nope, q_pe], dim=-1)
        
        # 处理键值投影
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # 合并K的两个部分并扩展头维度
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        
        # 更新KV缓存
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v
        
        # 计算注意力分数 Q@K^T / sqrt(d_k)
        scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        
        # 应用因果mask（训练时）
        if mask is not None:
            scores += mask.unsqueeze(1)  # 广播mask到所有头
        
        # Softmax归一化
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # 注意力加权求和
        x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        
        # 投影回模型维度
        x = self.wo(x.flatten(2))
        return x

class MLP(nn.Module):
    """
    多层感知机（MLP），用作前馈神经网络层

    属性:
        w1 (nn.Linear): 输入层到隐藏层的线性变换
        w2 (nn.Linear): 隐藏层到输出层的线性变换
        w3 (nn.Linear): 并行计算的特征变换层（门控分支）
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        初始化MLP层

        参数:
            dim: 输入和输出的特征维度
            inter_dim: 隐藏层的特征维度
        """
        super().__init__()
        # 第一层线性变换：dim -> inter_dim
        self.w1 = nn.Linear(dim, inter_dim)
        # 第二层线性变换：inter_dim -> dim（恢复原始维度）
        self.w2 = nn.Linear(inter_dim, dim)
        # 并行线性变换层：dim -> inter_dim（用于门控计算）
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程：
        1. 使用SILU激活函数处理w1的输出
        2. 与w3的输出进行逐元素相乘（门控机制）
        3. 通过w2进行最终变换

        参数:
            x: 输入张量，形状为(batch_size, seq_len, dim)

        返回:
            输出张量，形状与输入相同
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    """
    混合专家模型（MoE）中的门控路由机制
    
    参数:
        dim (int): 输入特征的维度
        topk (int): 每个输入激活的top专家数
        n_groups (int): 路由分组数
        topk_groups (int): 输入路由的目标组数
        score_func (str): 评分函数（'softmax' 或 'sigmoid'）
        route_scale (float): 路由权重的缩放因子
        weight (torch.nn.Parameter): 可学习的门控权重
        bias (Optional[torch.nn.Parameter]): 可选的门控偏置项
    """
    def __init__(self, args: ModelArgs):
        """
        初始化门控模块

        参数:
            args: 包含模型配置参数的ModelArgs对象
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts  # 每个样本激活的专家数
        self.n_groups = args.n_expert_groups  # 专家分组数量
        self.topk_groups = args.n_limited_groups  # 每个样本选择的分组数
        self.route_scale = args.route_scale  # 路由权重缩放因子
        
        # 路由权重矩阵：形状为(专家数量, 特征维度)
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # 特殊情况下（维度为7168）添加偏置项
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播过程：
        1. 计算路由分数
        2. 分组选择与掩码处理
        3. 选择topk专家并计算路由权重

        参数:
            x: 输入张量，形状为(batch_size, dim)

        返回:
            (weights, indices): 路由权重和专家索引
        """
        # 计算原始路由分数 [batch_size, n_experts]
        scores = F.linear(x, self.weight)
        scores = scores.softmax(dim=-1, dtype=torch.float32)  # 标准化为概率分布
        original_scores = scores  # 保存原始分数用于后续计算
        
        # 添加偏置项（如果存在）
        if self.bias is not None:
            scores = scores + self.bias
        
        # 分组路由逻辑
        if self.n_groups > 1:
            # 将分数重塑为 [batch_size, n_groups, group_size]
            scores = scores.view(x.size(0), self.n_groups, -1)
            
            # 计算每个组的得分：当没有偏置时取最大值，有偏置时取top2的和
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            
            # 选择得分最高的topk_groups个组
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            
            # 创建掩码：将未选中的组标记为-inf
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        # 选择topk专家
        indices = torch.topk(scores, self.topk, dim=-1)[1]  # [batch_size, topk]
        
        # 从原始分数中收集路由权重，并应用缩放因子
        weights = original_scores.gather(1, indices)  # [batch_size, topk]
        weights *= self.route_scale  # 应用路由缩放
        
        return weights.type_as(x), indices  # 保持数据类型一致

class MoE(nn.Module):
    """
    混合专家模型（Mixture of Experts）模块
    
    参数:
        dim (int): 输入特征的维度
        n_routed_experts (int): 模型中的专家总数
        n_local_experts (int): 分布式系统中本地处理的专家数
        n_activated_experts (int): 每个输入激活的专家数
        gate (nn.Module): 路由输入到专家的门控机制
        experts (nn.ModuleList): 专家模块列表
        shared_experts (nn.Module): 应用于所有输入的共享专家
    """
    def __init__(self, args: ModelArgs):
        """
        初始化MoE模块

        参数:
            args (ModelArgs): 包含MoE参数的模型参数
        """
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts  # 总专家数量
        self.n_local_experts = args.n_routed_experts  # 本地专家数量
        self.n_activated_experts = args.n_activated_experts  # 激活的专家数
        
        # 专家索引范围（用于分布式训练）
        self.experts_start_idx = 0
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        
        # 初始化门控模块
        self.gate = Gate(args)
        
        # 初始化专家列表（使用MLP作为专家网络）
        self.experts = nn.ModuleList([
            MLP(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        
        # 共享专家处理所有输入
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程：
        1. 通过门控获取路由权重和专家索引
        2. 稀疏激活选中的专家进行计算
        3. 叠加共享专家的计算结果

        参数:
            x: 输入张量，形状为(batch_size, seq_len, dim)

        返回:
            输出张量，形状与输入相同
        """
        original_shape = x.size()
        x = x.view(-1, self.dim)  # 展平为二维张量 [batch*seq_len, dim]
        
        # 获取路由权重和专家索引
        weights, indices = self.gate(x)  # weights: [batch*seq_len, topk], indices: [batch*seq_len, topk]
        
        # 初始化输出张量
        y = torch.zeros_like(x)
        
        # 统计每个专家被选中的次数（用于负载均衡）
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
        # 遍历本地专家进行计算
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:  # 跳过未被选中的专家
                continue
                
            expert = self.experts[i]  # 获取对应的专家模块
            idx, top_pos = torch.where(indices == i)  # 找到选择该专家的样本索引
            
            # 加权计算结果并累加到输出
            y[idx] += expert(x[idx]) * weights[idx, top_pos, None]  # [n_samples, dim]
        
        # 共享专家处理所有输入
        z = self.shared_experts(x)  # [batch*seq_len, dim]
        
        # 合并结果并恢复原始形状
        return (y + z).view(original_shape)

class Block(nn.Module):
    """
    Transformer模块块，包含注意力层和前馈层

    属性:
        attn: 多头注意力层
        ffn: 前馈网络（普通MLP或MoE）
        attn_norm: 注意力层的归一化
        ffn_norm: 前馈层的归一化
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        参数:
            layer_id: 当前层的索引（决定使用普通MLP还是MoE）
            args: 模型配置参数
        """
        super().__init__()
        self.attn = MLA(args)  # 多头注意力层（假设MLA已定义）
        
        # 前n_dense_layers层使用普通MLP，后续层使用MoE
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        
        # 初始化RMSNorm层
        self.attn_norm = nn.RMSNorm(args.dim)
        self.ffn_norm = nn.RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        前向传播流程：
        1. 注意力残差连接
        2. 前馈网络残差连接

        参数:
            x: 输入张量 [batch_size, seq_len, dim]
            start_pos: 序列起始位置（用于旋转位置编码）
            freqs_cis: 预计算的旋转位置编码
            mask: 注意力掩码

        返回:
            输出张量，形状与输入相同
        """
        # 注意力残差连接
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        # 前馈网络残差连接
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    """
    包含位置嵌入、多层结构和输出投影的Transformer模型
    
    参数:
        max_seq_len (int): Transformer的最大序列长度
        embed (nn.Module): 输入令牌的嵌入层
        layers (torch.nn.ModuleList): Transformer块列表
        norm (nn.Module): 所有块后的层归一化
        head (nn.Module): 映射到词表大小的输出投影层
        freqs_cis (torch.Tensor): 预计算的旋转嵌入复数值
    """
    def __init__(self, args: ModelArgs):
        """
        初始化Transformer模型

        参数:
            args (ModelArgs): 包含Transformer参数的模型参数
        """
        super().__init__()
        self.max_seq_len = args.max_seq_len
        
        # 词嵌入层
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        
        # 堆叠Transformer模块
        self.layers = nn.ModuleList([
            Block(layer_id, args) for layer_id in range(args.n_layers)
        ])
        
        # 最终归一化层
        self.norm = nn.RMSNorm(args.dim)
        
        # 输出投影层（词表大小）
        self.head = nn.Linear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        
        # 注册预计算的位置编码（非持久化缓冲区）
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        完整前向传播流程：
        1. 词嵌入
        2. 逐层处理Transformer模块
        3. 最终归一化和投影

        参数:
            tokens: 输入token IDs [batch_size, seq_len]
            start_pos: 序列起始位置（用于增量解码）

        返回:
            logits: 输出logits [batch_size, seq_len, vocab_size]
        """
        seqlen = tokens.size(1)
        
        # 词嵌入 [batch_size, seq_len, dim]
        h = self.embed(tokens)
        
        # 获取当前位置编码
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        # 生成因果注意力掩码（仅当seqlen>1时）
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = mask.triu_(diagonal=1)  # 上三角掩码
        
        # 逐层处理
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        
        # 最终归一化
        h = self.norm(h)
        
        # 输出投影
        logits = self.head(h)
        return logits

if __name__ == "__main__":
    # torch.set_default_dtype(torch.bfloat16)
    # torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(x.shape, model(x).size())
