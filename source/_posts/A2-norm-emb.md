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
#### Task 1: 均方根层归一化 (RMS Norm)

均方根层归一化（RMS Norm）是深度学习中应用最广泛的归一化模块，尤其在自然语言处理（NLP）和大语言模型（LLM）领域。该模块以形状为 `[batch_size, seqlen, hidden_size]` 的张量为输入（记为 `X`，形状为 `[b, s, h]`），并沿着隐藏层 `h` 维度，执行带可学习缩放变换的均方根归一化操作，得到输出 `Y`，形状为 `[b, s, h]`。具体公式如下所示：

$$
Y = \frac{X}{RMS[X]} \odot \gamma
$$

$$
RMS[X]=\sqrt{\frac{1}{h} \sum_{i=1}^{h}x_i^2 + \epsilon}
$$

其中，$RMS[X]$ 表示 `X` 的均方根，对于 `i in batch_size` 且 `j in seqlen`，对每一个 `X[i][j]`（形状为 `[hidden_size, ]`），独立地计算 *RMS*；$\epsilon$ 是一个极小的常数，用于避免除数为0，记作 `eps`；$\gamma$ 是沿 `h` 维度的可学习参数矩阵，直接与所有 `batch_size` 和 `seqlen` 的隐藏层做 *Hadamard* 乘积，若 `X` 的形状为 `[b, s, h]`，则 $\gamma$ 应该是一个形状为 `[1, 1, h]` 的参数矩阵。

为了将上述均方根层归一化泛化，在 **Task1** 中我们将实现上述模块的一个变体，称为分组均方根层归一化（**Group RMSNorm**）。给定分组大小 `group size`，简记为 `gz`，将 `X`  和 $\gamma$ 的隐藏层维度 `h` 均匀划分为 `Xg` 组，并对第 `i` 组分别应用 $(1) (2)$ 式中的 *RMS Norm* 操作，具体公式如下：
$$
Y_{g_i}=\frac{X_{g_i}}{RMS[X_{g_i}]} \odot \gamma_{g_i}
$$

$$
RMS[X_{g_i}]=\sqrt{\frac{1}{gz} \sum_{j=1}^{gz}x_{g_i, j}^2 + \epsilon}
$$

此外，我们还应该为该 *Group RMS Norm* 模块实现一个名为 `reset_parameters` 的参数初始化方法，用于为可学习的参数矩阵 $\gamma$ 设置初始值。我们会提供一个随机数种子（记为 `init_seed`，如42）和一个初始值范围元祖（记为 `init_range`，如 `(-1, 1)`），请使用均匀分布（**uniform distribution**）和 *pytorch* 自带的初始化方法为 *Parameter* 初始化。 

##### TODO

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

#### Task 2: 嵌入词表 (Vocab Embedding)

在 **Task2** 中，我们将要实现一个嵌入词表，以获取之前任务中的输入 `X`。假设词表的大小为 `vocab_size`，简记为 `v`，嵌入词表模块以形状为 `[batch_size, seqlen]` 的张量 `I` 作为输入，张量 `I` 中存储了每个 token 的 ID，ID 的范围是 `[0, v-1]`。通过查询可学习的嵌入表（记为 `T`，形状为 `[v, e]`），为张量 `I` 中的每个 ID 分配对应的嵌入向量，并返回形状为 `[batch_size, seqlen, emb_size]` 的嵌入张量 `E`，简记为 `[b, s, e]`。

与 **Task1** 类似，你还应该为 `VocabEmbedding` 模块类实现 `reset_parameters` 方法，用于嵌入表 `T` 的初始化。选用正态分布（**normal distribution**），给定平均值（表示为 `init_mean`，如 `0.`），标准差（表示为 `init_std`，如 `1.`），以及随机数种子（表示为 `init_seed`，如 `42`），对嵌入表 `T` 初始化，`reset_parameters` 方法同样需要在 `__init__` 中显示调用。

##### TODO

完成 `src/modeling/vocab_emb.py` 中的 `VocabEmbedding` 模块，实现上述嵌入词表。首先，你需要根据 `init_mean`, `init_std` 和 `init_seed`，使用 **normal distribution** 对嵌入表 `T` 初始化，然后将 `I` 作为输入，实现词表嵌入，并返回嵌入张量 `E`。

{% note warning %}

1. 输入 `I` 存储每个 token 的 ID，其 `dtype` 为 `torch.long`。
2. 你的实现不应该更改 `I`，包括 `I` 的数值与属性（包括 `I` 的 `shape`， `dtype` 和 `device` 等），因为 `I` 可能还有其他用途。
3. 参数中的 `dtype` 和 `device` 仅针对可学习嵌入表 `T` ，`T` 的 `dtype` 和 `device` 可能与 `I` 的不同，可以使用这两个参数和 `torch.nn.Parameter` 完成对 `T` 的申请与初始化。
4. 返回的嵌入张量 `E` 的 `device` 应与 `I` 相同，`dtype` 与 `T` 相同。

{% endnote %}

#### Task 3: 分布式并行嵌入词表 (Parallel Vocab Embedding)

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

##### TODO

完成 `src/modeling/vocab_emb.py` 中的 `ParallelVocabEmbedding` 模块，实现上述嵌入词表。首先，你需要根据 `init_mean`, `init_std` 和 `init_base_seed`，使用 **normal distribution** 对嵌入表 `Tr` 初始化，然后将 `I` 作为输入，实现词表嵌入，并返回不完整的嵌入张量 `Er`。

{% note warning %}

1. 输入 `I` 存储每个 token 的 ID，其 `dtype` 为 `torch.long`。
2. 你的实现不应该更改 `I`，包括 `I` 的数值与属性（包括 `I` 的 `shape`， `dtype` 和 `device` 等），因为 `I` 可能还有其他用途。
3. 参数中的 `dtype` 和 `device` 仅针对可学习嵌入表 `Tr` ，`Tr` 的 `dtype` 和 `device` 可能与 `I` 的不同，可以使用这两个参数和 `torch.nn.Parameter` 完成对 `Tr` 的申请与初始化。
4. 返回的嵌入张量 `Er` 的 `device` 应与 `I` 相同，`dtype` 与 `Tr` 相同。
5. 在所有测试用例中，`v` 均能被 `w` 整除，但仍然建议在 `__init__` 方法中使用 `assert` 进行检查，并附上错误提示，这是编程的良好习惯。

{% endnote %}