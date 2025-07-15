---
title: A4 Attention Module
date: 2025-07-13 11:01:59
tags:
  - Attention
  - Mask
categories:
  - Assignment
comments: false
mathjax: true
---

对于本次作业，我们将继续 Modeling 任务，以帮助你更深入地理解 Transformer 的各个组成模块。本次将特别关注 Transformer 结构核心的关键层之一：**Attention Layer（注意力层）**。

# Task 1: Offline Sliding-Window Attention

**Multi-head Attention（多头注意力机制）** 模块是 Transformer 中一个至关重要的构建单元（具体内容可见参考文献），接收三个张量作为输入：
- Query tensor（查询张量），记作 $\mathbf{Q}$，满足 $\mathbf{Q} \in \mathbb{R}^{\text{batch\_size} \times \text{seq\_len\_q} \times \text{num\_head\_q} \times \text{head\_dim}}$，记为 `[b, sq, hq, hd]`。
- Key tensor（键张量） 和 Value tensor（值张量），记为$\mathbf{K}, \mathbf{V}$，两者具有相同的形状，满足 $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{\text{batch\_size} \times \text{seq\_len\_kv} \times \text{num\_head\_kv} \times \text{head\_dim}}$，记为 `[b, skv, hkv, hd]`。

$\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 共同构成了 Attention 模块的输入，用于后续的注意力计算。注意，对于 Multi-head Attention，batch\_size 和 num\_head 都应该被看作是 **batch-like** 维，而对于 `seq_len` 维度，$\mathbf{Q}$ 中的每一张量 $\mathbf{q}_\text{i}$ 可以被看作是第 i 个 `token` 的潜在嵌入查询信息（embedded latent query message），用于从知识 $\mathbf{V}$ 中查询相关信息，而 $\mathbf{V}$ 中的每一张量 $\mathbf{v}_\text{j}$ 可以被看作是第 j 个 `token` 的潜在嵌入知识表示（embedded latent knowledge archive）。为了聚合 $\mathbf{V}$ 中所有重要的信息并忽略其他无关的信息，每个 $\mathbf{v}_\text{j}$ 都对应一个潜在嵌入关键词张量 $\mathbf{k}_\text{j}$。通过计算 $\mathbf{q}_\text{i}$ 和 $\mathbf{k}_\text{j}$ 的点积标量 $\mathbf{q}_\text{i}\mathbf{k}_\text{j}^\top$，我们可以得到关于 $\mathbf{q}_\text{i}$ 和 $\mathbf{v}_\text{j}$ 之间的**相似度得分（similarity score）**。最终，每个 $\mathbf{q}_\text{i}$ 对应的聚合结果 $\mathbf{o}_\text{i}$ 表示为对 $\mathbf{V}$ 中所有 $\mathbf{v}_\text{j}$ 的加权和，即 $\mathbf{o}_\text{i} := \sum\limits_j \mathbf{a}^{(i)}_j\mathbf{v}_\text{j}$，其中，权重向量 $\mathbf{a}^{(i)}$ 由上述每个查询 $\mathbf{q}_\text{i}$ 与所有关键词 $\mathbf{k}_\text{j}$ 的 **归一化点积相似度（normalized dot-product similarity）** 构成。至于权重的归一化方式，最常见的做法是应用 `softmax` 操作，这被称为一种“软最大化（soft maximalization）”操作。其目的是让模型只 **关注（pay attention to）** 那些真正重要、即相似度得分最高的信息。

因此，整个 Attention 操作（针对每个 `batch` 和每个 `head`）可以表示为：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{A} \times \mathbf{V}
$$

$$
\text{where} \space \mathbf{A} = \text{softmax}_{\text{row-wise}}(\text{scale} · \mathbf{P}) \in \mathbb{R}^{\text{sq} \times \text{skv}}
$$

$$
\space \mathbf{P} = \mathbf{Q} \times \mathbf{K}^\top + \mathbf{M}\in \mathbb{R}^{\text{sq} \times \text{skv}}
$$

其中，上式中 $\mathbf{M}$ 用于实现注意力机制中的**掩码机制**，$\mathbf{M}$ 是一个二值 `Attention Mask`，其每个元素的取值为 $-\infty$ 或 $0$，用于在计算注意力时进行筛选：
- 若某一对 ($\mathbf{q}_\text{i}$, $\mathbf{k}_\text{j}$) 是无关的，则对应位置的 `Mask` 值为 $−\infty$，从而在 `softmax` 中被强制置接近 $0$（即被“屏蔽”）；
- 若该对是相关的，则对应位置的 `Mask` 值为 $0$，保留其注意力得分。

常见的掩码模式包括：
- 全量掩码（Full Mask）：每个 `token` 可以关注所有 `token`，如图 a；
- 因果掩码（Causal Mask）：每个 `token` 只能关注它之前的 `token` 及其自身。即 $\mathbf{q}_\text{i}$ 只能最多关注 $\mathbf{k}_\text{j}$ 满足 $\text{j} \leq \text{i}$ 的情况，不能看到未来的信息，如图 b；
- 滑动窗口掩码（Sliding Window Mask）：每个 `token` 只能关注窗口内的 `token`：
  - 对于 `Full` 场景， $\mathbf{q}_\text{i}$ 只能最多关注 $\mathbf{k}_\text{j}$ 满足 $\text{j} \in [i-w, i+w]$，如图 c；
  - 对于 `Causal` 场景， $\mathbf{q}_\text{i}$ 只能最多关注 $\mathbf{k}_\text{j}$ 满足 $\text{j} \in [i-w, i]$，如图 d。

![mask](mask.svg)

另外，由于 `softmax` 操作对数值变化非常敏感，一般会采取一些策略来稳定其计算过程。最常见的做法是对 `softmax` 的输入 $\mathbf{P}$ 进行缩放处理，即 $\text{scale} · \mathbf{P}$，其中，`scale` 通常设置为 $\frac{1}{\sqrt{hd}}$，以防止当维度增大时，数值激增而导致梯度不稳定。最近，Nvidia 在其论文中还引入了一些额外的技巧，用于在训练过程中进一步提升 `softmax` 操作的稳定性（详见参考文献中的 Nvidia 论文），我们也将采用其中的一些方法，具体包括：

1. **Softmax Temperature（温度系数）**：为了控制 `softmax` 分布的尖锐程度（sharpness），我们可以对 `softmax` 输入 $\mathbf{P}$ 应用温度系数，形式为：$\frac{\mathbf{P}}{temp}$。其中，`temp` 是一个取值范围在 $(0, +\infty)$ 的超参数，通过调节 temperature，可以控制模型对高相似度的响应程度，是调节注意力权重敏感度的重要手段：
   - 当 $\text{temp} = 1.0$ 时，`softmax` 分布是原始分布；
   - 当 $\text{temp} \to 0.0$ 时，`softmax` 分布变得更加尖锐（sharp），即更接近 `one-hot`；
   - 当 $\text{temp} \to +\infty$ 时，`softmax` 分布则变得更加平滑（smooth），各项概率更接近均匀分布。
2. **Softmax Capping（上限截断）**：除了使用 `softmax temperature` 外，我们还可以通过 `softmax capping` 来自适应地控制 $\mathbf{P}$ 的数值范围,形式为：$\text{cap} \cdot \text{tanh}(\frac{\mathbf{P}}{\text{cap}})$。其中，`cap` 通常是一个较大的正数，这种方法的作用类似于一个自适应版本的 `softmax temperature`：
   - 当 $\mathbf{P}$ 较小时，输出几乎不变；
   - 当 $\mathbf{P}$ 较大时，使用 `tanh` 对其进行平滑限制，防止极端值导致梯度不稳定；
   - 由于 `softmax capping` 和 `softmax temperature` 都是为了调控 `softmax` 的数值稳定性，因此在一次前向传播中，我们只使用其中一个。
3. **Softmax Clipping（剪裁）**：为了抑制 `Attention` 权重 $\mathbf{A}$ 中的异常值（outliers）过大增长，我们可以对 $\mathbf{A}$ 应用 `softmax clipping`，具体操作为：$\mathbf{A}_{\text{clipped}} = \text{clip} \left( (r - l) \cdot \mathbf{A} + l,\ 0,\ 1 \right)$，这种方法在不破坏归一化的前提下，有效地限制了 `Attention` 分布中的离群值，增强数值稳定性并降低过拟合风险，其中：
   - $\mathbf{A}$ 是原始的 `softmax` 输出，数值范围在 $[0, 1]$；
   - $[l, r]$ 是一个扩展范围（super-range），满足 $l \leq 0，r \geq 1$；
   - 这一步操作先将 $\mathbf{A}$ 从 $[0, 1]$ 线性映射到 $[l, r]$ 区间，再 **clip（裁剪）** 回 $[0, 1]$，从而截断极端值。
4. **Softmax Dropout（注意力丢弃）**：为了提升注意力权重 $\mathbf{A}$ 的鲁棒性（robustness），我们可以对 $\mathbf{A}$ 应用 `softmax dropout`，形式为：$\mathbf{A}_{\text{dropout}} = \text{dropout}_p(\mathbf{A})$，其中：
   - $p \in [0, 1]$ 是 `dropout rate`（丢弃率）；
   - 该操作会随机将 $\mathbf{A}$ 中的部分权重置为 0，并相应地对其余部分进行缩放，以保持总和不变。
5. **QK 层归一化（QK Layer Normalization）**：为了进一步缓解 $\mathbf{P}$ 中可能出现的过大数值问题（这可能导致注意力权重 $\mathbf{A}$ 退化为近似 `one-hot` 形式），我们可以选择对 $\mathbf{Q}$ 和 $\mathbf{K}$ 预先应用 `Layer Normalization`（层归一化）。在本次作业中，我们将传统的 `Layer Normalization` 替换为 `Group RMS Normalization`（组 RMS 归一化），以充分利用我们在 **A2** 中实现的 `GroupRMSNorm` 模块。

最终，整个 `OfflineSlidingWindowAttn` 操作（针对每个 `batch` 和每个 `head`）可以表示为：

$$
\text{OfflineSlidingWindowAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{\widehat A} \times \mathbf{V}
$$

$$
\text{where} \space \mathbf{\widehat A} = \text{dropout}\space_p(\text{clip}((r-l) \mathbf{\tilde A} + l, 0, 1))
$$

$$
\space \mathbf{\tilde A} = \text{softmax}\space_{\text{row-wise}}(\mathbf{\tilde P})
$$

$$
\space \mathbf{\tilde P} = \begin{cases}
\cfrac{\text{scale} \cdot \mathbf{\tilde Q} \times \mathbf{\tilde K}^\top}{\text{temp}} + \mathbf{M}_{\text{sw}} \space(+ \mathbf{M}_{\text{causal}}), & \text{softmax temperature} \\
\text{cap}\cdot \text{tanh}(\cfrac{\text{scale} \cdot \mathbf{\tilde Q} \times \mathbf{\tilde K}^\top}{\text{cap}}) + \mathbf{M}_{\text{sw}} \space(+ \mathbf{M}_{\text{causal}}), & \text{softmax capping} \\
\end{cases}
$$

$$
\text{where}\space \mathbf{\tilde Q} = \text{GroupRMSNorm}(\mathbf{Q}), \space \mathbf{\tilde K} = \text{GroupRMSNorm}(\mathbf{K})
$$

同时，为了使 `OfflineSlidingWindowAttn` 模块更灵活地适应不同格式的输入，我们在 `src/modeling/attention.py` 中定义了一个枚举类 `AttnQKVPackFormat`，用于定义 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 输入张量的打包方式：
- **AttnQKVPackFormat.Q_K_V**：最常见的格式，其中 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 是三个独立的张量；
- **AttnQKVPackFormat.Q_KV**：在这种格式下，$\mathbf{K}$、$\mathbf{V}$ 沿着 `num_heads` 维度被打包在一起，构成一个张量，而 $\mathbf{Q}$ 仍然是单独的张量；
- **AttnQKVPackFormat.QKV**：此格式下，$\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 都沿 `num_heads` 维度打包成一个张量，这种情况下，$\mathbf{Q}$ 和 $\mathbf{K}$、$\mathbf{V}$ 的其他维度（如序列长度、batch 大小等）必须相同，以保证解包后的结构正确。

另外，我们还在 `src/modeling/attention.py` 中设计了另一个枚举类 `AttnQKVLayout`，用于定义 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 张量的形状布局（shape layout），以支持不同的输入格式：
- **AttnQKVLayout.BSHD**：最常见的布局形式，满足 $\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{\text{batch\_size} \times \text{seq\_len\_kv} \times \text{num\_head\_kv} \times \text{head\_dim}}$ ，记为 `bshd`；
- **AttnQKVLayout.SBHD**：更适用于**分布式环境（distributed environment）** 的布局，满足 $\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{\text{seq\_len\_kv} \times \text{batch\_size} \times \text{num\_head\_kv} \times \text{head\_dim}}$ ，记为 `sbhd`；
- **AttnQKVLayout.THD**：最通用的布局格式，也称为 `varlen layout`（变长布局），满足 $\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{\text{total\_seq\_len} \times \text{num\_head\_kv} \times \text{head\_dim}}$，在这种布局中，不存在显式的 `batch` 维度，所有长度不一的序列会沿着 `sequence` 维度拼接在一起，在这种情况下需要额外提供两个辅助输入，用于标识每条序列在拼接后的张量中的位置：
  - `cu_seqlens_q`；
  - `cu_seqlens_kv`；
  这两个张量都是 `int32` 类型，形状为 `[batch_size + 1]`，其中每一段 `[[cu_seqlens}[i], cu_seqlens[i+1]]` 表示第 i 个样本在 $\mathbf{Q}$ 或 $\mathbf{K}$、$\mathbf{V}$ 中的 起止区间（start-end），在 `varlen layout` 场景的掩码模式可参考下图。（更多示例请参考 Flash Attention 接口中的相关内容。）

![sliding window](window.svg)

## TODO

**完成 `src/modeling/attention.py` 中的 `OfflineSlidingWindowAttn` 模块**，实现上述注意力机制运算，具体细节包括：
- 参数中的 `dtype` 和 `device` 是用于 `GroupRMSNorm` 中可学习参数的，它们可能与 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 的 `dtype` 和 `device` 不同。
- 返回的输出张量 $\mathbf{O}$ 的元属性（meta attributes），包括 `dtype`、`device` 和 `layout`，应当与 $\mathbf{Q}$ 保持一致。
- 只有当参数 `softmax_cap` 被设置为 `None` 时，才可以使用 `softmax_temp` 参数启用 `softmax` 温度策略（softmax temperature strategy）。
- 所有参数都会被保证处于其合法范围内。
- $\mathbf{Q}$ 和 $\mathbf{K}$ 的 `GroupRMSNorm` 是 `OfflineSlidingWindowAttn` 模块中的**独立子层（individual sub-layers）**，因为 `GroupRMSNorm` 只接受形状为 `[batch_size, seq_len, hidden_size]` 的三维张量，而其中的 `hidden_size = num_heads * head_dim`，在 $\mathbf{Q}$ 和 $\mathbf{K}$ 之间可能不同。此外，我们确保 `head_dim` 可以被 `group_size` 整除，即不会存在某个 `group` 在 `hidden` 维度上跨越两个不同 `head` 的情况。
- 当 `num_heads` 在 $\mathbf{Q}$ 和 $\mathbf{K}$、$\mathbf{V}$ 之间不相同时（即 `MQA` 或 `GQA` 风格，满足 `num_q_head != num_kv_head` 且 `num_q_head % num_kv_head == 0`，详见参考文献中的相关论文），我们采用相同的 `kv-heads` 重复策略（kv-heads repeating strategy）来使 $\mathbf{Q}$ 与 $\mathbf{K}$、$\mathbf{V}$ 在 `head` 数上保持一致。（参见参考文献中的 Llama Attention Layer 和 PyTorch 的 **repeat_interleave** 函数了解更多细节）。
- 当 $\mathbf{Q}$ 和 $\mathbf{K}$、$\mathbf{V}$ 在 `sequence` 维度上不一致时（例如在 `cross-attention` 或 `autoregressive decoding` 阶段），`attention mask M` 就不是一个方阵，而是形状为 `[sq, skv]` 的长方形矩阵，也可以看作是一个从完整 `Attention` 方阵 `[max(sq, skv), max(sq, skv)]` 中“滑动”出的窗口（slide）。**注意**：此时对于 `Causal` 掩码场景，我们应该从这个完整的 `Attention` 方阵中选取哪一块长方形窗口？对于 `OfflineSlidingWindowAttn` 模块，我们选择对齐完整方阵右下角区域（即 **bottom-right** 的掩码模式），遵循 flash-attention 的设置，参考下图，。（详见参考文献中的 Flash Attention 接口了解更多示例。）

<img src="bottom-right.svg" style="width: 30%; height: auto;">

## Offline Sliding-Window Attention 小结

总结来说，你需要实现 `OfflineSlidingWindowAttn` 模块。该模块接收以不同打包格式（`packing formats`）和不同布局（`layouts`）表示的 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 作为输入（如果布局为 `AttnQKVLayout.THD`，则需额外提供 `cu_seqlens_q` 和 `cu_seqlens_kv`），执行上述所描述的 `offline sliding window attention` 操作，并返回一个与 $\mathbf{Q}$ 使用相同布局的输出张量 $\mathbf{O}$。


