"""模型定义模块：LSTM 基线、普通 Transformer（消融对照）与 Informer 主模型。

把模型类集中在这里而不是散落在各脚本里，是为了让训练脚本与
``05_evaluate_compare.py`` 共享同一份架构定义 —— 否则两处各写一份，
改一处忘一处，加载权重时形状对不上。

Informer 参考论文：Zhou et al., "Informer: Beyond Efficient Transformer
for Long Sequence Time-Series Forecasting", AAAI 2021。
本实现忠实于论文的三个核心机制（ProbSparse 注意力 / 自注意力蒸馏 /
生成式解码器），并在注释中标明每个简化点的理由。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM 基线模型（与 03_baseline_lstm.py 中的定义完全一致）。

    架构：LSTM(2 层, hidden=128) → 取最后时刻隐状态 → Linear → 未来 24 步。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_len: int = 24,
        dropout: float = 0.2,
    ) -> None:
        """构造 LSTM 基线模型。

        Args:
            input_size: 特征维数。
            hidden_size: 隐层维度。
            num_layers: LSTM 层数。
            output_len: 预测步数。
            dropout: 层间 dropout。
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """取最后时刻隐状态做多步预测。

        Args:
            x: 形状 ``(N, seq_len, input_size)``。

        Returns:
            形状 ``(N, output_len)`` 的预测。
        """
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TransformerModel(nn.Module):
    """时序 Transformer 主模型。

    架构：输入投影 → 正弦位置编码 → TransformerEncoder × N → 聚合 → 输出头。

    两个刻意的小设计：
        1. **正弦位置编码**而非可学习编码：训练样本只有约 2000 个，
           可学习编码增加参数量却几乎无收益；正弦编码零参数且
           天然支持序列长度外推。
        2. **池化可配置**（mean / last / concat）：光伏功率夜间恒为 0，
           最后一步的隐状态可能在夜间全零窗口上信息稀疏；平均池化
           让整个窗口的信息参与预测，concat 则同时保留两种视角。
           默认 mean，最优配置由超参扫描确定（见 04 脚本注释）。

    Args:
        input_size: 特征维数（本项目为 12）。
        d_model: 内部表示维度。
        nhead: 多头注意力头数，必须整除 ``d_model``。
        num_layers: Encoder 层数。
        dim_feedforward: 前馈网络维度。
        dropout: Dropout 比例。
        output_len: 预测步数（本项目为 24）。
        pool: ``"mean"``（平均池化）、``"last"``（取最后一步）或
            ``"concat"``（mean 与 last 拼接，信息最全）。
        max_seq_len: 位置编码表的最大长度。
    """

    def __init__(
        self,
        input_size: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_len: int = 24,
        pool: str = "mean",
        max_seq_len: int = 128,
    ) -> None:
        """构造时序 Transformer 模型。

        Args:
            input_size: 特征维数（本项目为 12）。
            d_model: 内部表示维度。
            nhead: 多头注意力头数，必须整除 ``d_model``。
            num_layers: Encoder 层数。
            dim_feedforward: 前馈网络维度。
            dropout: Dropout 比例。
            output_len: 预测步数（本项目为 24）。
            pool: ``"mean"``（平均池化）、``"last"``（取最后一步）或
                ``"concat"``（mean 与 last 拼接，信息最全）。
            max_seq_len: 位置编码表的最大长度。
        """
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")
        if pool not in ("mean", "last", "concat"):
            raise ValueError(f"pool 必须是 'mean'/'last'/'concat'，实际为 {pool!r}")

        self.pool = pool
        self.input_proj = nn.Linear(input_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        head_in = d_model * 2 if self.pool == "concat" else d_model
        self.head = nn.Linear(head_in, output_len)

        # 正弦位置编码注册为 buffer：随模型保存/加载，但不算可训练参数
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe.unsqueeze(0))  # (1, max_seq, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 形状 ``(N, seq_len, input_size)``。

        Returns:
            形状 ``(N, output_len)`` 的预测。
        """
        seq_len = x.size(1)
        if seq_len > self.pos_encoding.size(1):
            raise ValueError(
                f"输入序列长度 {seq_len} 超过位置编码表长度 "
                f"{self.pos_encoding.size(1)}，请调大 max_seq_len"
            )

        x = self.input_proj(x)  # (N, seq, d_model)
        x = x + self.pos_encoding[:, :seq_len, :]  # 加位置信息
        x = self.encoder(x)  # (N, seq, d_model)

        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "last":
            x = x[:, -1, :]
        else:  # concat：mean 与 last 拼接，两种视角互补
            x = torch.cat([x.mean(dim=1), x[:, -1, :]], dim=1)
        return self.head(x)  # (N, output_len)


# ======================================================================
# Informer：ProbSparse 自注意力 + 自注意力蒸馏 + 生成式解码器
# ======================================================================


class ProbSparseAttention(nn.Module):
    """ProbSparse 自注意力 —— Informer 的核心机制（论文第 3.1 节）。

    动机：
        普通自注意力的复杂度是 O(L²)，且长序列上每个 query 的注意力
        分布接近均匀（只有少数 query 与某些 key 强相关，其余是"懒惰"的）。
        Informer 提出：只对少数"活跃" query 计算完整注意力，
        其余 query 直接取 value 的均值即可，误差可控、速度大幅提升。

    稀疏度度量（论文公式 4）：
        M(qi, K) = max_j{qi·kjᵀ/√d} − (1/L_K)·Σ_j{qi·kjᵀ/√d}

    含义：注意力分布越"尖锐"（与某几个 key 强相关），M 越大。
    实现上先对每个 query 采样 U_part 个 key 近似计算 M，
    选出 top-u 个活跃 query 计算全量注意力，其余输出 value 均值。

    Args:
        factor: 稀疏采样因子。u = factor·ln(L_Q)，U_part = factor·ln(L_K)。
        dropout: 注意力权重 dropout。
    """

    def __init__(self, factor: int = 5, dropout: float = 0.1) -> None:
        """构造 ProbSparse 注意力。

        Args:
            factor: 稀疏采样因子。
            dropout: 注意力权重 dropout。
        """
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def _prob_qk(self, q: torch.Tensor, k: torch.Tensor, sample_k: int, n_top: int):
        """计算活跃 query 的注意力分数。

        Args:
            q: ``(B, H, L_Q, D)``。
            k: ``(B, H, L_K, D)``。
            sample_k: 采样 key 数量。
            n_top: 活跃 query 数量。

        Returns:
            ``(qk, M_top)``：top-u query 对全部 key 的分数，
            以及活跃 query 在 L_Q 维上的索引。
        """
        b, h, l_k, d = k.shape

        # 均匀采样 key 子序列，近似计算稀疏度 M（论文公式 4）
        idx = torch.linspace(0, l_k - 1, sample_k).long().to(q.device)
        k_sample = k[:, :, idx, :]  # (B,H,sample_k,D)
        qk_sample = torch.matmul(q, k_sample.transpose(-2, -1))
        m_score = qk_sample.max(dim=-1)[0] - qk_sample.mean(dim=-1)

        # 取 M 最大的 top-u 个 query 的索引
        m_top = m_score.topk(n_top, sorted=False)[1]  # (B,H,u)

        # 按索引收集活跃 query，对全部 key 计算完整分数
        idx_expand = m_top.unsqueeze(-1).expand(-1, -1, -1, d)
        q_reduce = torch.gather(q, dim=2, index=idx_expand)
        qk = torch.matmul(q_reduce, k.transpose(-2, -1))  # (B,H,u,L_K)
        return qk, m_top

    def _initial_context(self, v: torch.Tensor, l_q: int) -> torch.Tensor:
        """非活跃 query 的兜底输出：value 的均值（论文的 mean(V)）。"""
        mean_v = v.mean(dim=-2, keepdim=True)  # (B,H,1,D)
        return mean_v.expand(-1, -1, l_q, -1).clone()

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """ProbSparse 注意力前向。

        Args:
            queries: ``(B, H, L_Q, D)``。
            keys: ``(B, H, L_K, D)``。
            values: ``(B, H, L_V, D)``（L_V == L_K）。

        Returns:
            ``(B, H, L_Q, D)`` 的注意力输出。

        Note:
            本实现不处理 attention mask —— 解码器序列很短（48 步），
            那里用普通因果注意力即可（见 :class:`InformerDecoderLayer`
            的说明），掩码版 ProbSparse 的收益与复杂度不成比例。
        """
        b, h, l_q, d = queries.shape
        _, _, l_k, _ = keys.shape

        u = min(int(self.factor * math.ceil(math.log(max(l_q, 2)))), l_q)
        u_part = min(int(self.factor * math.ceil(math.log(max(l_k, 2)))), l_k)

        qk, m_top = self._prob_qk(queries, keys, u_part, u)
        scale = d**-0.5
        attn = torch.softmax(qk * scale, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, values)  # (B,H,u,D)

        # 活跃 query 回填到原位置，其余位置用 value 均值
        out = self._initial_context(values, l_q)
        idx_expand = m_top.unsqueeze(-1).expand_as(context)
        out.scatter_(dim=2, index=idx_expand, src=context)
        return out


class InformerAttentionLayer(nn.Module):
    """ProbSparse 多头注意力封装：线性投影 → 分头 → ProbSparse → 输出投影。"""

    def __init__(self, d_model: int, nhead: int, factor: int = 5, dropout: float = 0.1) -> None:
        """构造 ProbSparse 多头注意力层。

        Args:
            d_model: 内部表示维度。
            nhead: 注意力头数。
            factor: ProbSparse 采样因子。
            dropout: 注意力权重 dropout。
        """
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ProbSparseAttention(factor=factor, dropout=dropout)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """输入均为 ``(B, L, d_model)``，输出同形。"""
        b, l_q, _ = queries.shape
        l_k = keys.shape[1]

        def _heads(x: torch.Tensor, length: int) -> torch.Tensor:
            return x.reshape(b, length, self.nhead, self.head_dim).transpose(1, 2)

        q = _heads(self.q_proj(queries), l_q)
        k = _heads(self.k_proj(keys), l_k)
        v = _heads(self.v_proj(values), l_k)

        out = self.attention(q, k, v)  # (B,H,L_Q,head_dim)
        out = out.transpose(1, 2).reshape(b, l_q, -1)
        return self.out_proj(out)


class ConvDistill(nn.Module):
    """自注意力蒸馏层（论文第 3.2 节）。

    通过 1D 卷积 + 最大池化把序列长度减半：
        - 提取相邻时间步的局部依赖（注意力层后的特征平滑）；
        - 序列减半让上层注意力只关注"重要"的粗粒度表示，
          同时把内存开销降到 O((2−ε)L·logL)。
    """

    def __init__(self, d_model: int) -> None:
        """构造蒸馏层。

        Args:
            d_model: 内部表示维度（卷积通道数）。
        """
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 ``(B, L, d_model)``，输出 ``(B, L//2, d_model)``。"""
        x = x.transpose(1, 2)  # (B, d, L) 卷积按通道
        x = self.pool(self.activation(self.conv(x)))
        return x.transpose(1, 2)


class InformerEncoderLayer(nn.Module):
    """Informer 编码层：ProbSparse 自注意力 + 前馈网络（Pre-LN 结构）。"""

    def __init__(
        self,
        attention: InformerAttentionLayer,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """构造 Informer 编码层。

        Args:
            attention: ProbSparse 多头注意力层。
            d_model: 内部表示维度。
            d_ff: 前馈网络中间维度。
            dropout: 残差连接的 dropout。
            activation: 前馈网络激活函数。
        """
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码层前向：自注意力与前馈各带残差与 LayerNorm（Pre-LN）。

        Args:
            x: ``(B, L, d_model)`` 的输入序列。

        Returns:
            同形状的输出。
        """
        new_x = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(new_x))
        y = self.ff(x)
        return self.norm2(x + self.dropout(y))


class InformerEncoder(nn.Module):
    """Informer 编码器：e_layers 个编码层，除最后一层外每层后接蒸馏。"""

    def __init__(self, layers: list[InformerEncoderLayer], convs: list[ConvDistill | None]) -> None:
        """构造编码器。

        Args:
            layers: 编码层列表。
            convs: 与各层对应的蒸馏层列表，``None`` 表示该层后不蒸馏。
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.convs = nn.ModuleList([c for c in convs if c is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码器前向：逐层编码，除最后一层外每层后接蒸馏。

        Args:
            x: ``(B, L, d_model)`` 的输入序列。

        Returns:
            ``(B, L_mem, d_model)`` 的记忆序列，
            L_mem 取决于层数与蒸馏（96 → 48 → 24）。
        """
        conv_idx = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 and conv_idx < len(self.convs):
                x = self.convs[conv_idx](x)
                conv_idx += 1
        return x


class InformerDecoderLayer(nn.Module):
    """Informer 解码层：因果自注意力 + 与编码器记忆的交叉注意力。

    简化说明（诚实记录）：
        论文与官方实现里，解码器自注意力也是掩码版 ProbSparse。
        本项目的解码器序列只有 label_len + pred_len = 48 步，
        O(L²) 开销可忽略，且掩码 + top-u 采样的组合实现复杂、易错，
        因此这里用普通因果注意力。交叉注意力同样用全注意力 ——
        论文第 3.1 节的消融也表明解码器的稀疏化收益很小。
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """构造解码层。

        Args:
            self_attention: 因果自注意力层。
            cross_attention: 与编码器记忆的交叉注意力层。
            d_model: 内部表示维度。
            d_ff: 前馈网络中间维度。
            dropout: 残差连接的 dropout。
            activation: 前馈网络激活函数。
        """
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """解码层前向：自注意力 → 交叉注意力 → 前馈，各带残差与 LayerNorm。

        Args:
            x: 解码序列 ``(B, label_len + pred_len, d_model)``。
            memory: 编码器记忆 ``(B, L_mem, d_model)``。
            mask: 因果掩码 ``(L, L)``，下三角为 1。

        Returns:
            同形状的输出。
        """
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, memory, memory)))
        return self.norm3(x + self.dropout(self.ff(x)))


class FullAttentionLayer(nn.Module):
    """普通多头注意力（解码器与交叉注意力使用）。"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        """构造普通多头注意力层。

        Args:
            d_model: 内部表示维度。
            nhead: 注意力头数。
            dropout: 注意力权重 dropout。
        """
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """多头注意力前向。

        Args:
            queries: ``(B, L_Q, d_model)`` 的查询。
            keys: ``(B, L_K, d_model)`` 的键。
            values: ``(B, L_K, d_model)`` 的值。
            mask: 可选掩码 ``(L_Q, L_K)``，0 的位置被屏蔽为 -inf。

        Returns:
            ``(B, L_Q, d_model)`` 的输出。
        """
        b, l_q, _ = queries.shape
        l_k = keys.shape[1]

        def _heads(x: torch.Tensor, length: int) -> torch.Tensor:
            return x.reshape(b, length, self.nhead, self.head_dim).transpose(1, 2)

        q = _heads(self.q_proj(queries), l_q)
        k = _heads(self.k_proj(keys), l_k)
        v = _heads(self.v_proj(values), l_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, l_q, -1)
        return self.out_proj(out)


def generate_sinusoidal(max_seq_len: int, d_model: int) -> torch.Tensor:
    """生成正弦位置编码表，形状 ``(1, max_seq_len, d_model)``。"""
    position = torch.arange(max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class InformerModel(nn.Module):
    """Informer 主模型：稀疏编码器 + 生成式解码器（论文第 3 节）。

    前向流程：
        1. 编码器输入完整历史窗口（本项目 96 步 × 12 特征），
           经 ProbSparse 注意力与蒸馏后得到记忆序列；
        2. 解码器输入"开始标记"（历史窗口的最后 label_len 步）+
           全零占位符（pred_len 步），先做因果自注意力，再交叉关注
           编码器记忆 —— 即论文的"一次前向生成全部预测"风格；
        3. 取解码输出的最后 pred_len 个位置，线性投影到标量功率。

    Args:
        input_size: 特征维数（本项目为 12）。
        d_model: 内部表示维度。
        nhead: 多头注意力头数，必须整除 ``d_model``。
        e_layers: 编码层数。
        d_layers: 解码层数。
        d_ff: 前馈网络维度。
        dropout: Dropout 比例。
        factor: ProbSparse 采样因子。
        label_len: 解码器"开始标记"长度。
        pred_len: 预测步数（本项目为 24）。
        distil: 是否启用编码器蒸馏。
        activation: 激活函数。
    """

    def __init__(
        self,
        input_size: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        e_layers: int = 3,
        d_layers: int = 1,
        d_ff: int = 128,
        dropout: float = 0.1,
        factor: int = 5,
        label_len: int = 24,
        pred_len: int = 24,
        distil: bool = True,
        activation: str = "gelu",
    ) -> None:
        """构造 Informer 主模型。

        Args:
            input_size: 特征维数（本项目为 12）。
            d_model: 内部表示维度。
            nhead: 多头注意力头数，必须整除 ``d_model``。
            e_layers: 编码层数。
            d_layers: 解码层数。
            d_ff: 前馈网络中间维度。
            dropout: Dropout 比例。
            factor: ProbSparse 采样因子。
            label_len: 解码器"开始标记"长度。
            pred_len: 预测步数（本项目为 24）。
            distil: 是否启用编码器蒸馏。
            activation: 前馈网络激活函数。
        """
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 nhead({nhead}) 整除")

        self.label_len = label_len
        self.pred_len = pred_len
        self.distil = distil

        self.enc_embedding = nn.Linear(input_size, d_model)
        self.dec_embedding = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(generate_sinusoidal(256, d_model), requires_grad=False)

        # 编码器：e_layers 层，除最后一层外每层后接蒸馏
        encoder_layers = []
        distill_layers: list[ConvDistill | None] = []
        for i in range(e_layers):
            encoder_layers.append(
                InformerEncoderLayer(
                    InformerAttentionLayer(d_model, nhead, factor, dropout),
                    d_model,
                    d_ff,
                    dropout,
                    activation,
                )
            )
            # 最后一层不蒸馏（保留最大细粒度），其余每层后蒸馏
            distill_layers.append(ConvDistill(d_model) if (distil and i < e_layers - 1) else None)
        self.encoder = InformerEncoder(encoder_layers, distill_layers)

        # 解码器：d_layers 层（因果自注意力 + 交叉注意力）
        self.decoder = nn.ModuleList(
            [
                InformerDecoderLayer(
                    FullAttentionLayer(d_model, nhead, dropout),
                    FullAttentionLayer(d_model, nhead, dropout),
                    d_model,
                    d_ff,
                    dropout,
                    activation,
                )
                for _ in range(d_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, 1)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x_enc: 编码器输入 ``(B, L_enc, input_size)``。
            x_dec: 解码器开始标记 ``(B, label_len, input_size)``；
                ``None`` 时自动取 ``x_enc`` 的最后 ``label_len`` 步。

        Returns:
            形状 ``(B, label_len + pred_len)`` 的完整解码输出。
            前 ``label_len`` 个位置对应开始标记的重构（teacher forcing 用），
            后 ``pred_len`` 个位置是未来预测，调用方取 ``[:, -pred_len:]``。
        """
        b, l_enc, _ = x_enc.shape
        device = x_enc.device

        if x_dec is None:
            x_dec = x_enc[:, -self.label_len :, :]
        label_len = x_dec.shape[1]

        # ---- 编码器 ----
        enc = self.enc_embedding(x_enc) + self.pos_encoding[:, :l_enc, :]
        memory = self.encoder(enc)

        # ---- 解码器：开始标记 + 全零占位符 ----
        dec = self.dec_embedding(x_dec)
        total_len = label_len + self.pred_len
        dec = torch.cat(
            [dec, torch.zeros(b, self.pred_len, dec.shape[-1], device=device)],
            dim=1,
        )
        # 拼接后统一加一次位置编码（避免开始标记部分重复相加）
        dec = dec + self.pos_encoding[:, :total_len, :]

        # 因果掩码：位置 i 只能看到 ≤ i（下三角为 1）
        causal = torch.tril(torch.ones(total_len, total_len, device=device))

        for layer in self.decoder:
            dec = layer(dec, memory, mask=causal)

        # 输出整个解码序列（与论文一致）：label 部分由训练脚本用已知真实值
        # 做 teacher forcing，pred 部分才是真正要用的预测。
        # 因此返回 (B, label_len + pred_len)，脚本取最后 pred_len 个位置。
        return self.projection(self.norm(dec))[..., 0]
