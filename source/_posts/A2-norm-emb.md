---
title: A2 RMSNorm and Embedding
date: 2025-06-17 14:17:05
tags:
  - RMSNorm
  - Vocab Embedding
categories:
  - Assignment
comments: false
mathjax: true
---
在本次实验中，我们将实现 *Transfomer* 中的两个关键模块：均方根层归一化（**RMSNorm**）和词嵌入（**Embedding**）层。**RMSNorm** 是一种高效的归一化方法，通过仅利用输入的均方根值进行归一化，在保持模型性能的前提下，提升了计算效率与训练稳定性。**Embedding** 层则广泛应用于自然语言处理任务，其主要作用是将离散的符号（如单词或子词）映射为稠密的向量表示，从而为模型提供可学习的语义基础。

# Task 1: 均方根层归一化 (RMS Norm)

均方根层归一化（RMS Norm）是深度学习中应用最广泛的归一化模块，尤其在自然语言处理（NLP）和大语言模型（LLM）领域。该模块以形状为 `[batch_size, seqlen, hidden_size]` 的张量为输入（记为 `X`，形状为 `[b, s, h]`），并沿着隐藏层 `h` 维度，执行带可学习缩放变换的均方根归一化操作，得到输出 `Y`，形状为 `[b, s, h]`。具体公式如下所示：
$$
Y=\frac{X}{RMS[X]} \odot \gamma \tag{1}
$$

$$
RMS[X]=\sqrt{\frac{1}{h} \sum_{i=1}^{h}x_i^2 + \epsilon} \tag{2}
$$

其中，$RMS[X]$ 表示 `X` 的均方根，对于 `i in batch_size` 且 `j in seqlen`，对每一个 `X[i][j]`（形状为 `[hidden_size, ]`），独立地计算 *RMS*；$\epsilon$ 是一个极小的常数，用于避免除数为0，记作 `eps`；$\gamma$ 是沿 `h` 维度的可学习参数矩阵，直接与所有 `batch_size` 和 `seqlen` 的隐藏层做 *Hadamard* 乘积，若 `X` 的形状为 `[b, s, h]`，则 $\gamma$ 应该是一个形状为 `[1, 1, h]` 的参数矩阵。

为了将上述均方根层归一化泛化，在 **Task1** 中我们将实现上述模块的一个变体，称为分组均方根层归一化（**Group RMSNorm**）。给定分组大小 `group size`，简记为 `gz`，将 `X`  和 $\gamma$ 的隐藏层维度 `h` 均匀划分为 `Xg` 组，并对第 `i` 组分别应用 $(1) (2)$ 式中的 *RMS Norm* 操作，具体公式如下：
$$
Y_{g_i}=\frac{X_{g_i}}{RMS[X_{g_i}]} \odot \gamma_{g_i} \tag{3}
$$

$$
RMS[X_{g_i}]=\sqrt{\frac{1}{gz} \sum_{j=1}^{gz}x_{g_i, j}^2 + \epsilon} \tag{4}
$$

此外，我们还应该为该 *Group RMS Norm* 模块实现一个名为 `reset_parameters` 的参数初始化方法，用于为可学习的参数矩阵 $\gamma$ 设置初始值。我们会提供一个随机数种子（记为 `init_seed`，如42）和一个初始值范围元祖（记为 `init_range`，如 `(-1, 1)`），请使用均匀分布（**uniform distribution**）和 *pytorch* 自带的初始化方法为 *Parameter* 初始化。 

## TODO

完成 `src/modeling/norm.py` 中的 `GroupRMSNorm` 模块，实现上述参数初始化和分组均方根归一化。首先，你需要根据 `init_range` 和 `init_seed` ，使用 **uniform distribution** 为 $\gamma$ 初始化，然后将 `X` 和 `gz` 作为输入，实现**Group RMSNorm**，并返回输出 `Y`，形状为 `[batch_size, seqlen, hidden_size]`。

{% note warning %}

1. 参数中的 `dtype` 和 `device` 仅针对可学习参数矩阵 $\gamma$ ，$\gamma$ 的 `dtype` 和 `device` 可能与 `X` 的不同，可以使用这两个参数和 `torch.nn.Parameter` 完成对 $\gamma$ 的申请与初始化。
2. 输出 `Y` 的属性（包括 `dtype` 和 `device`）必须与输入 `X` 保持一致。
3. 由于均方根归一化（*RMS Norm*）涉及除法计算，建议使用 `float32` 等高精度数据类型以保持数值稳定。
4. 在所有测试用例中，`h` 均能被 `gz` 整除，但仍然建议在 `__init__` 方法中使用 `assert` 进行检查，并附上错误提示，这是编程的良好习惯。
5. 初始化参数时，`reset_parameters` 方法应在 `__init__` 方法中调用一次。

{% endnote %}

{% note info %}

请自行查阅 `pytorch` 中乘法的广播机制，对 **Task1** 的实现有很大帮助。

{% endnote %}

# Task 2: 嵌入词表 (Vocab Embedding)

在 **Task2** 中，我们将要实现一个嵌入词表，以获取之前任务中的输入 `X`。假设词表的大小为 `vocab_size`，简记为 `v`，嵌入词表模块以形状为 `[batch_size, seqlen]` 的张量 `I` 作为输入，张量 `I` 中存储了每个 token 的 ID，ID 的范围是 `[0, v-1]`。通过查询可学习的嵌入表（记为 `T`，形状为 `[v, e]`），为张量 `I` 中的每个 ID 分配对应的嵌入向量，并返回形状为 `[batch_size, seqlen, emb_size]` 的嵌入张量 `E`，简记为 `[b, s, e]`。

与 **Task1** 类似，你还应该为 `VocabEmbedding` 模块类实现 `reset_parameters` 方法，用于嵌入表 `T` 的初始化。选用正态分布（**normal distribution**），给定平均值（表示为 `init_mean`，如 `0.`），标准差（表示为 `init_std`，如 `1.`），以及随机数种子（表示为 `init_seed`，如 `42`），对嵌入表 `T` 初始化，`reset_parameters` 方法同样需要在 `__init__` 中显示调用。

## TODO

完成 `src/modeling/vocab_emb.py` 中的 `VocabEmbedding` 模块，实现上述嵌入词表。首先，你需要根据 `init_mean`, `init_std` 和 `init_seed`，使用 **normal distribution** 对嵌入表 `T` 初始化，然后将 `I` 作为输入，实现词表嵌入，并返回嵌入张量 `E`。

{% note warning %}

1. 输入 `I` 存储每个 token 的 ID，其 `dtype` 为 `torch.long`。
2. 你的实现不应该更改 `I`，包括 `I` 的数值与属性（包括 `I` 的 `shape`， `dtype` 和 `device` 等），因为 `I` 可能还有其他用途。
3. 参数中的 `dtype` 和 `device` 仅针对可学习嵌入表 `T` ，`T` 的 `dtype` 和 `device` 可能与 `I` 的不同，可以使用这两个参数和 `torch.nn.Parameter` 完成对 `T` 的申请与初始化。
4. 返回的嵌入张量 `E` 的 `device` 应与 `I` 相同，`dtype` 与 `T` 相同。

{% endnote %}

# Task 3: 分布式并行嵌入词表 (Parallel Vocab Embedding)

在 **Task3** 中，我们将在 **Task2** 实现的嵌入词表的基础上，实现分布式的嵌入词表。随着 **LLM** 规模迅速扩大，词表的大小已经增长到 `128K+`，嵌入词表很难在一块 **GPU**上存储和计算。

因此，我们将实现一个“分布式并行嵌入词表”模块解决这个问题。假设通信组的大小为 `world_size`，简记为 `w`，在本实验中你可以简单的理解为 **GPU** 的数量，每块 **GPU** 都会有一个序号 `rank`（记为 `r`，且 $r \in[0,w-1]$），我们将大小为 `v` 的词表均匀的分配到 `w` 张 **GPU** 中，每张卡获取大小为 `v//w` 的一个分片。通过这种方式，可以减小单卡 **GPU** 中嵌入表的存储压力，还能并行执行词表嵌入，以加速计算。

{% note info %}

在真实的分布式环境中，`world_size` 和 `rank` 都可以直接从环境变量和通信组中获取，但限于资源有限，我们省去通信，仅保留计算逻辑，并直接在参数中给出 `world_size` 和 `rank`，以模拟分布式环境。

{% endnote %}

给定词表大小 `v`，嵌入维度 `e`，**GPU** 序号 `r`，**GPU** 数量 `w`，并行词表嵌入模块的流程如下：

1. 对于序号为 `r` 的 **GPU**，分得大小为 `n = v // m` 的词表，其只关注区间 $[r \cdot n, (r+1)\cdot n-1]$ 内的词元 ID，该区间记为 `R`；
2. 从正态分布中初始化局部嵌入表 `Tr`，请自行计算 `Tr` 的形状；
3. 接收输入张量 `I`，对其中属于区间 `R` 的 ID 查询 `Tr` 获取嵌入向量，对超出范围的 ID 用全零向量替代；
4. 计算得到局部嵌入 `Er`，形状与标准嵌入 `E` 一致，但仅包含区间 `R` 内 ID 有效的嵌入向量，其余位置为全零。（通过通信累加所有 **GPU** 的 `Er` 即可重构完整词表的嵌入结果，本实验省去通信累加步骤）

与 **Task2** 类似，你还应该为 `ParallelVocabEmbedding` 模块类实现 `reset_parameters` 方法，用于嵌入表 `Tr` 的初始化。不同的是，此时参数中的随机数种子是基础随机数种子，记为 `init_base_seed`，而真正的随机数种子应为 `init_base_seed + r`，以避免对所有的参数矩阵进行相同的初始化。

## TODO

完成 `src/modeling/vocab_emb.py` 中的 `ParallelVocabEmbedding` 模块，实现上述嵌入词表。首先，你需要根据 `init_mean`, `init_std` 和 `init_base_seed`，使用 **normal distribution** 对嵌入表 `Tr` 初始化，然后将 `I` 作为输入，实现词表嵌入，并返回不完整的嵌入张量 `Er`。

{% note warning %}

1. 输入 `I` 存储每个 token 的 ID，其 `dtype` 为 `torch.long`。
2. 你的实现不应该更改 `I`，包括 `I` 的数值与属性（包括 `I` 的 `shape`， `dtype` 和 `device` 等），因为 `I` 可能还有其他用途。
3. 参数中的 `dtype` 和 `device` 仅针对可学习嵌入表 `Tr` ，`Tr` 的 `dtype` 和 `device` 可能与 `I` 的不同，可以使用这两个参数和 `torch.nn.Parameter` 完成对 `Tr` 的申请与初始化。
4. 返回的嵌入张量 `Er` 的 `device` 应与 `I` 相同，`dtype` 与 `Tr` 相同。
5. 在所有测试用例中，`v` 均能被 `w` 整除，但仍然建议在 `__init__` 方法中使用 `assert` 进行检查，并附上错误提示，这是编程的良好习惯。

{% endnote %}

# [Optional] Task4：旋转位置编码

Transformer 模型将输入的词元（token）视为一个“词袋”并并行处理，因而本身不具备对序列顺序的感知能力。为保留输入中的序列信息，最初版本的 Transformer 引入了一种新颖的正弦位置编码（Sinusoidal Positional Encoding，简称 SinPE），其定义如下面公式所示：
$$
\text{SinPE}(n) :=
\begin{bmatrix}
\sin{\left(n\theta^0\right)} \cr
\cos{\left(n\theta^0\right)} \cr
\sin{\left(n\theta^1\right)} \cr
\cos{\left(n\theta^1\right)} \cr
\vdots \cr
\sin\left(n\theta^{\frac{d}{2}-1}\right) \cr
\cos\left(n\theta^{\frac{d}{2}-1}\right)
\end{bmatrix}
\quad \text{where }
\theta := \beta^{-1},\
\beta := \text{base}^{\frac{2}{d}},\
n \in \{0, 1, \ldots, L - 1\}
\tag{5}
$$
其中，`L` 表示序列长度，`d` 表示隐藏层维度，`base` 是一个人为设定的大整数，通常取值为10000（请参考原始论文），$\beta$ 是三角函数基的波长或周期的幂次基数，随着维度 `i` 的增大而按几何级数增长，其形式为 $\beta ^ i$，其中 $i=0,1,\ldots,\frac{d}{2}-1$。

相比之下，旋转位置编码（Rotary Position Embedding，简称 RoPE）在处理长序列时提供了更稳定的方案。它在具备绝对位置信息感知能力的同时，能够捕捉相对位置模式，因此被广泛应用于当前的主流开源大模型（如 LLaMA，ChatCLM）中。随着研究的推进，RoPE 逐渐取代了原始的 SinPE、可学习位置编码（Learnable PE）以及相对位置编码（Relative PE），成为当前 Transformer 结构中位置编码的主流选择。

更具体的说，RoPE 在复数域中对隐藏状态进行旋转操作，而不像 SinPE 那样将位置编码加到隐藏状态中。该方法与 SinPE 共享相同的基函数，如下式所示：
$$
\text{RoPE}(n) := 
\begin{bmatrix}
R_n^{(0)} \cr
\phantom{R_n^{(0)}}& R_n^{(1)} \cr
\phantom{R_n^{(0)}}& \phantom{R_n^{(0)}}& \ddots \cr
\phantom{R_n^{(0)}}&\phantom{R_n^{(0)}}&\phantom{R_n^{(0)}}& R_n^{\left(\frac{d}{2} - 1\right)}
\end{bmatrix},
\quad \text{where } 
R_n^{(i)} := 
\begin{bmatrix}
\cos(n\theta^i) & -\sin(n\theta^i) \cr
\sin(n\theta^i) & \cos(n\theta^i)
\end{bmatrix}
\tag{6}
$$
尽管 RoPE（旋转位置编码）具备相对距离衰减和训练稳定性等优势，但在序列长度的外推能力方面仍然存在不足，尤其是在“短序列训练、长序列推理”（Train Short and Test Long）场景下表现不佳（详见参考文献中的 Length Extrapolation 相关论文）。因此，已有多项研究致力于扩展 RoPE 的泛化能力，使其在推理时能有效处理远超训练长度的序列。

在这些方法中，**NTK-aware RoPE** 通过结合高频外推和低频内插来提升外推性能。它通过缩放系数 $c_𝜅$ 对参数 $\beta$ 进行调整，从而实现在最低频率项上以比例 $𝜅$ 进行等效插值，同时保持高频项的尺度不变，如下式所示。这种非线性缩放方式可以直接应用于使用 RoPE 预训练的大语言模型（如 Llama），无需微调即可扩展其上下文长度的边界，这一方法已被 *CodeLlama* 所采纳（详见参考文献中的 Llama RoPE 源代码）。

$$
\tilde{\beta} := c_\kappa \cdot \beta, \quad 
s.t. \quad \frac{n}{\tilde{\beta}^{d/2 - 1}} = \frac{n/\kappa}{\beta^{d/2 - 1}} 
\Rightarrow c_\kappa = \kappa^{2/(d - 2)} \tag{7}
$$
在 **Task4** 中，你需要像 `Llama` 一样实现 `NTKAwareRoPE` 模块，但是，有一些差异如下：

- 标准的 RoPE 模块在前向传播时仅返回余弦/正弦基张量，形状为 `[seqlen, head_dim]`，该参数对记作 `(C, S)`，分别存储 $\sin{n\theta^i}$ 和  $\cos{n\theta^i}$（请参考 $(5)(6)$ 式中对 $n,\theta$ 的定义）。实际的旋转编码操作是在另一个独立的函数 `apply_rotary_pos_emb` 完成。
- 我们遵循这种设计模式：你需要在 `src/functional.py` 中实现 `apply_rotary_pos_emb` 函数，该函数会在 `src/modeling/pos_emb.py` 中导入，并在 `NTKAwareRoPE` 的 `forward` 方法中被调用。与标准做法不同的是，`NTKAwareRoPE` 的 `forward` 方法不仅返回 `(C, S)` 的基张量，还应对输入张量 `X` 应用旋转编码并返回嵌入后的输出张量 `E`，其中：
  - 输入张量 `X` 的形状为 `[batch_size, seqlen, num_heads, head_dim]`，记作 `[b, s, nh, hd]`；
  - 输出张量 `E` 的形状与 `X` 的形状相同，表示应用旋转编码后的结果。

- 由于 **RoPE** 矩阵的稀疏性，直接用矩阵乘法来实现会很浪费算力，推荐使用 $(8)$ 中的方法来实现标准 **RoPE**。其中，$\bigotimes$ 是逐位相乘，并且可以注意到 $(8)$ 中，$X$ 按 $(x_0,x_1),(x_2,x_3),\cdots$ 的顺序进行分组。而 `Llama` 和 `ChatGLM` 的 **RoPE** 模块对 $X$ 的顺序进行了重组，也即按 $(x_0,x_{\frac{d}{2}}),(x_1,x_{\frac{d}{2}+1}),\cdots$ 的顺序进行分组，请使用 $(9)$ 中的方法实现 **Task4** 中的 **RoPE**。

  {% note info%}

  余弦\正弦参数对 `(C, S)` 也要做相应调整。

  {% endnote %}

$$
\begin{pmatrix}
x_0 \vphantom{\cos{\left(n\theta^0\right)}}\cr
x_1 \vphantom{\cos{\left(n\theta^0\right)}} \cr
x_2 \vphantom{\cos{\left(n\theta^0\right)}}\cr
x_3 \vphantom{\cos{\left(n\theta^0\right)}}\cr
\vdots \cr
x_{d-2} \vphantom{\cos\left(n\theta^{\frac{d}{2}-1}\right)}\cr
x_{d-1} \vphantom{\cos\left(n\theta^{\frac{d}{2}-1}\right)}
\end{pmatrix}
\quad
\bigotimes
\quad
\begin{pmatrix}
\cos{\left(n\theta^0\right)} \cr
\cos{\left(n\theta^0\right)} \cr
\cos{\left(n\theta^1\right)} \cr
\cos{\left(n\theta^1\right)} \cr
\vdots \cr
\cos\left(n\theta^{\frac{d}{2}-1}\right) \cr
\cos\left(n\theta^{\frac{d}{2}-1}\right)
\end{pmatrix}
\quad
+
\quad
\begin{pmatrix}
-x_1 \vphantom{\cos{\left(n\theta^0\right)}}\cr
x_0 \vphantom{\cos{\left(n\theta^0\right)}} \cr
-x_3 \vphantom{\cos{\left(n\theta^0\right)}}\cr
x_2 \vphantom{\cos{\left(n\theta^0\right)}}\cr
\vdots \cr
-x_{d-1} \vphantom{\cos\left(n\theta^{\frac{d}{2}-1}\right)}\cr
x_{d-2} \vphantom{\cos\left(n\theta^{\frac{d}{2}-1}\right)}
\end{pmatrix}
\quad
\bigotimes
\quad
\begin{pmatrix}
\sin{\left(n\theta^0\right)} \cr
\sin{\left(n\theta^0\right)} \cr
\sin{\left(n\theta^1\right)} \cr
\sin{\left(n\theta^1\right)} \cr
\vdots \cr
\sin\left(n\theta^{\frac{d}{2}-1}\right) \cr
\sin\left(n\theta^{\frac{d}{2}-1}\right)
\end{pmatrix}
\tag{8}
$$


$$
\begin{pmatrix}
x_0 \vphantom{\cos{\left(n\theta^0\right)}}\cr
x_1 \vphantom{\cos{\left(n\theta^0\right)}} \cr
x_2 \vphantom{\cos{\left(n\theta^0\right)}}\cr
x_3 \vphantom{\cos{\left(n\theta^0\right)}}\cr
\vdots \cr
x_{\frac{d}{2}-1} \vphantom{\cos{\left(n\theta^0\right)}}\cr
x_{\frac{d}{2}} \vphantom{\cos{\left(n\theta^0\right)}} \cr
x_{\frac{d}{2}+1} \vphantom{\cos{\left(n\theta^0\right)}}\cr
x_{\frac{d}{2}+2} \vphantom{\cos{\left(n\theta^0\right)}}\cr
\vdots \cr
x_{d-2} \vphantom{\cos\left(n\theta^{\frac{d}{2}-1}\right)}\cr
x_{d-1} \vphantom{\cos\left(n\theta^{\frac{d}{2}-1}\right)}
\end{pmatrix}
\quad
\bigotimes
\quad
\begin{pmatrix}
\cos{\left(n\theta^0\right)} \cr
\cos{\left(n\theta^1\right)} \cr
\cos{\left(n\theta^2\right)} \cr
\cos{\left(n\theta^3\right)} \cr
\vdots \cr
\cos\left(n\theta^{\frac{d}{2}-1}\right) \cr
\cos{\left(n\theta^0\right)} \cr
\cos{\left(n\theta^1\right)} \cr
\cos{\left(n\theta^2\right)} \cr
\vdots \cr
\cos\left(n\theta^{\frac{d}{2}-2}\right) \cr
\cos\left(n\theta^{\frac{d}{2}-1}\right)
\end{pmatrix}
\quad
+
\quad
\begin{pmatrix}
-x_{\frac{d}{2}} \vphantom{\cos{\left(n\theta^0\right)}}\cr
-x_{\frac{d}{2}+1} \vphantom{\cos{\left(n\theta^0\right)}} \cr
-x_{\frac{d}{2}+2} \vphantom{\cos{\left(n\theta^0\right)}}\cr
-x_{\frac{d}{2}+3} \vphantom{\cos{\left(n\theta^0\right)}}\cr
\vdots \cr
-x_{d-1} \vphantom{\cos{\left(n\theta^0\right)}}\cr
x_0 \vphantom{\cos{\left(n\theta^0\right)}} \cr
x_1 \vphantom{\cos{\left(n\theta^0\right)}}\cr
x_2 \vphantom{\cos{\left(n\theta^0\right)}}\cr
\vdots \cr
x_{\frac{d}{2}-2} \vphantom{\cos\left(n\theta^{\frac{d}{2}-1}\right)}\cr
x_{\frac{d}{2}-1} \vphantom{\cos\left(n\theta^{\frac{d}{2}-1}\right)}
\end{pmatrix}
\quad
\bigotimes
\quad
\begin{pmatrix}
\sin{\left(n\theta^0\right)} \cr
\sin{\left(n\theta^1\right)} \cr
\sin{\left(n\theta^2\right)} \cr
\sin{\left(n\theta^3\right)} \cr
\vdots \cr
\sin\left(n\theta^{\frac{d}{2}-1}\right) \cr
\sin{\left(n\theta^0\right)} \cr
\sin{\left(n\theta^1\right)} \cr
\sin{\left(n\theta^2\right)} \cr
\vdots \cr
\sin\left(n\theta^{\frac{d}{2}-2}\right) \cr
\sin\left(n\theta^{\frac{d}{2}-1}\right)
\end{pmatrix}
\tag{9}
$$

- 另一个问题是，初始化 `NTKAwareRoPE` 时会提供一个训练阶段使用的最大序列长度（记作 `ms`）和一个缩放比例（记作 `k`，也即 $\kappa$），此时我们可以预先计算好 `(C, S)`，其形状为 `[es, hd]`，其中 `es = ms x k` 表示最大支持的拓展序列长度。因此，当有一个输入张量 `X_` 的实际序列长度 `s_` 超过了 `es`，即 `s_ > es`，我们必须动态重新计算一对新的 `(C_, S_)`，以确保旋转编码操作可以适用于这类超长输入。
- 但这里有两个问题：
  1. 当需要重新计算新的余弦/正弦基 `(C', S')` 时，我们应如何为输入张量 `X'` 确定新的缩放比例 `k'` ？
  2. 当遇到这类超长序列时，我们是否应该每次仅计算并使用该输入所需的 `(C', S')`，同时保留原始的缩放比例 `k` 及其对应的 `(C, S)` 用于常规输入？或者，我们应该每次都更新当前的 `k` 及其对应的 `(C, S)` 为新的 `k'` 和 `(C', S')` ？
- 上述问题尚无标准答案。在此任务中，我们采用如下策略：
  1. 当出现新的输入长度 `s' > es` 时，我们选择满足 `es' = ms x k' >= s'` 的最小 `k'`，其中 `k'` 是一个偶数；
  2. 我们在初始化 `NTKAwareRoPE` 模块时新增了一个参数 `dynamic`。当 `dynamic = True` 时，每次遇到超出长度的输入时，都会更新当前的 $k \leftarrow k'$ 以及 $(C,S) \leftarrow (C', S')$；反之，若 `dynamic = False` 时，则仅为当前超长输入临时计算并使用 $(C',S')$，而全局的 $k$ 和 $(C,S)$ 保持不变。 

## TODO

完成 `NTKAwareRoPe` 模块。该模块首先根据参数 `hd` , `ms`, `base`, `k` 初始化原始的位置编码参数对 `(C, S)`。接着，模块接收形状为`[b, s, nh, hd]`的输入张量`X`，并按以下逻辑处理：当序列长度 `s` 小于等于预设最大长度 `es` 时，直接调取缓存的 `(C, S)` 参数；若`s > es`，则重新计算出新的参数 `k_` ，并重新计算新的参数对 `(C_, S_)`。特别地，当参数 `dynamic` 设为 True 时，模块会在重新计算后同步更新内部存储的 `k` 值及 `(C, S)` 参数。最后，模块将通过调用需自行实现的 `apply_rotary_pos_emb` 函数，将对应位置的 `(C, S)` 参数应用于输入张量 `X` ，完成旋转位置编码操作并返回编码结果 `E` 。

{% note warning %}

1. 参数中的 `dtype` 和 `device` 仅针对位置编码参数对 `(C, S)`。通常我们需要更高的精度来处理位置嵌入，因此在所有测试用例中，我们会将数据类型固定为 `float32`，并且建议您在计算的每一步都使用 `float32` 以确保精度一致性。
2. 返回的张量 `E` 应与输入张量 `X` 保持相同的 `dtype` 和 `device`。
3. 在实际实现中，位置编码参数对 `(C, S)` 应被视为模块状态的一部分，不仅要能够随着模块一起迁移设备（例如通过 `module.to(device)` 方法），还应在保存模型状态字典时被忽略，因为它们可以根据需要轻松重构。因此，您不应将 `(C, S)` 作为普通 Python 属性直接赋值给 `self`，而是应将其注册为 PyTorch 的非持久缓冲区（Non-persistent Buffer）。具体操作请参考 PyTorch 文档中关于模块注册的相关内容。
4. 您可以参考 Llama 和 ChatGLM 等模型实现旋转位置编码（RoPE）的方式，但请特别注意上述要求，这些要求与 Llama 和 ChatGLM 的实现细节存在差异。

{% endnote %}

# References

*提示：以下是一些可能对你的任务有帮助的参考资料，或者可以加深/拓展你对深度学习中归一化层，词表嵌入层和位置编码的理解：*

**!! 请记住：查阅论文、源码以及官方文档，并从中进行思考和学习，是一项基本且至关重要的能力。请尽量不要过度依赖一些带有偏见或内容浅显的博客，例如 CSDN !!**

* [RMSNorm Paper](https://arxiv.org/abs/1910.07467)
* [Pytorch RMSNorm Module](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html#rmsnorm)
* [Llama RMSNorm Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L60)
* [ChatGLM RMSNorm Module](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L181)
* [Pytorch LayerNorm Module](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm)
* [Pytorch BatchNorm Module](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d)
* [Pytorch GroupNorm Module](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html#torch.nn.GroupNorm)
* [Pytorch Uniform Initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_)

* [Pytorch Embedding Module](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
* [Pytorch Embedding Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html)
* [ChatGLM Vocab Embedding Module](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L706)
* [Megatron Vovab Parallel Embedding Module](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py#L156)
* [Pytorch Normal Initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.normal_)

* `SinPE`: [paper](https://arxiv.org/abs/1706.03762) | [blog](https://spaces.ac.cn/archives/8231)
* `RoPE`: [paper](https://arxiv.org/abs/2104.09864) | [blog](https://spaces.ac.cn/archives/8265)
* `Length Extrapolation`: [Alibi](https://arxiv.org/abs/2108.12409) | [PI](https://arxiv.org/abs/2306.15595)
* `NTK-aware RoPE`: [blog](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) | [paper](https://arxiv.org/abs/2309.00071) | [survey](https://arxiv.org/abs/2311.12351)
* `Llama RoPE`:  [module](https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/llama/modeling_llama.py#L178) | [function](https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/llama/modeling_llama.py#L211)
* `ChatGLM RoPE`: [module](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L121) | [function](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L121)
* `Pytorch Module Register`: [buffer](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer) | [parameter](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_parameter)