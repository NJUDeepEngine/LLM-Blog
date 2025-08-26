---
title: A1 Matmul
date: 2025-06-14 12:57:11
tags:
  - Mutmal
  - Multi-head
categories:
  - Assignment
comments: false
mathjax: true
---
在本次实验中，我们将实现深度学习模型中非常核心且基础的运算模块——矩阵乘（Matrix Multiplication）。矩阵乘法不仅是构建神经网络中全连接层、卷积操作和注意力机制等模块的基础，同时也是高性能计算优化的重要对象。

# Task 1: MalMul with multi-head variant

在 task 1 中，我们要实现两个矩阵相乘的逻辑，我们有以下两个矩阵：

- `A1`：一个 3D 的输入张量，形状为 `[batch_size, seq_len, hidden_size]`，`batch_size` 表示序列的数量，`seqlen` 表示一个序列的最大长度，`hidden_size` 表示序列中每一个 `token` 拥有的维度。我们简写 `A1` 的形状为 `[b, s, h]`。
- `W1`：一个 2D 的权重张量，形状为 `[hidden_size, embed_size]`，它表示一个投影矩阵，将任何行向量从 `hidden_size`-dim 投影到 `embed_size`-dim。我们简写 `W1` 的形状为 `[h, e]`。

朴素的矩阵乘法仅对 `A1` 中 `batch_size` 维度，针对每个序列索引i，都执行 `O1[i] = A1[i] @ W1` 计算，从而得到形状为 `[b, s, e]` 的张量 `O1`。

在多头矩阵乘法中，我们首先将输入张量 `A1` 和权重张量 `W1` 的 `h` 维度均分为 `num_heads` 个子维度（记为 `nh`，表示头的数量），由此得到形状为 `[b, s, nh, hd]` 的四维张量 `A2` 和形状为 `[nh, hd, e]` 的三维张量 `W2`。接下来，对于 `A2` 中 `batch_size` 维度下的每个序列，遍历其 `num_heads` 维度上的每个 `[s, hd]` 矩阵，并将其与 W2 中 `num_heads` 维度下对应的 `[hd, e]` 矩阵进行乘法运算。通过多头并行计算，最终输出一个形状为 `[b, s, nh, e]` 的四维张量 `O2`。

## TODO

完成 `src/functional.py` 中的 `matmul_with_multi_head` 函数 ，实现上述多头矩阵乘法的逻辑，输入张量 `A1` 和 `W1`，返回计算值 `O2`。

{% note info %}
1. 输入的张量是 A1 和 W1，你需要自己将其转换为 A2 和 W2 再进行计算，请注意 torch 中 `reshape`, `view`, `transpose`, `permute`等函数的用法和区别。
2. 虽然逻辑上矩阵的乘法是用遍历进行计算的，但请勿使用 for 循环的方式进行实现，请自行查阅 pytorch 的计算函数，如 `@`, `torch.bmm` , `torch.mm` , `torch.matmul` , `torch.einsum` 等。
3. 了解并使用 pytorch 计算中的广播机制，有助于简化计算逻辑。
{% endnote %}

{% note warning %}
1. 所有输入张量均在同一设备（CPU 或 CUDA）上从标准正态分布 N (0, 1) 随机初始化，具有相同的数据类型（float32、float16 或 bfloat16），并且在所有测试用例中均未设置 `require_grad`；
2. 在所有测试用例中，`hidden_size` 均会被保证能被 `num_heads` 整除。
{% endnote %}

# Task 2: MalMul with importance

在多头矩阵乘法的基础上，我们引入一个表示“重要性”的概率张量 `P`，其形状为 `[b, s]`。P 中的每个元素表示 `A1` 中对应位置的元素的重要程度。基于这个重要性概率，我们的目标是只对每个序列中的 "重要" 元素执行矩阵乘法运算。这些重要元素总共有`total_important_seq_len` 个，简记为 `t`，其计算结果会被收集到输出张量 `O3` 中，其形状为 `[t, nh, e]`。

为了精确界定 "重要" 元素的范围，我们提供两个可选参数：

1. `top_p`：取值范围为 `[0., 1.]` 的浮点数。只有概率值大于或等于 `top_p` 的元素才被视为 "重要" 元素，默认值为 `1.0`。
2. `top_k`：取值范围为 `[1, ..., seq_len]` 的整数。对于批次中的每个序列，只将概率最高的 `top_k` 个元素视为 "重要" 元素。如果未设置 `top_k`（默认值为 `None`），则等价于 `top_k = seq_len`。

注意，必须同时满足上述两点的元素才是重要元素。

## TODO

完成 `src/functional.py` 中 `matmul_with_importance` 函数 **Task2** 的部分，实现上述重要性乘法。首先，你需要根据 `top_p` 和 `top_k` 的值，从 `A1` 中挑选出“重要”的元素，组成 `[t, h]` 的张量 `A3`，再仿造 **Task1** 中的多头矩阵乘法，输出 `[t, nh, e]` 的张量 `O3`。

{% note info %}
可以使用 `torch.topk` 计算 `topk` 个重要元素。
{% endnote %}

{% note warning %}
在所有测试用例中，`top_p` 和 `top_k` 参数均会被保证在各自有效范围内取值。
{% endnote %}

# Task 3: MalMul with grad

此外，如果提供了输出张量的可选梯度（记为 `dO3`，其形状与 `O3` 相同），我们还需要计算输入张量的梯度（记为 `dA1`，形状与 `A1` 相同）和权重张量的梯度（记为 `dW1`，形状与 `W1` 相同）。若未提供 `dO3`，则 `dA1` 和 `dW1` 均返回 `None`。

## TODO

完成 `src/functional.py/matmul_with_importance` 中 **Task3** 的部分，请参考 **A0** 中介绍的两种求梯度的方式，返回 `A1` 和 `W1` 的梯度。

{% note info %}
1. 若未提供 `grad_output` 参数，应避免计算梯度以提高效率并节省内存。
2. 若提供了 `grad_output` 参数，可使用 PyTorch 的自动求导机制计算梯度，但需注意潜在的副作用，这些副作用可能会在测试中被测试。
{% endnote %}

# References

*提示：以下是一些可能对你的任务有帮助的参考资料，或者可以加深/拓展你对 PyTorch 的理解：*

**!! 请记住：查阅论文、源码以及官方文档，并从中进行思考和学习，是一项基本且至关重要的能力。请尽量不要过度依赖一些带有偏见或内容浅显的博客，例如 CSDN !!**

* [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)
* [Pytorch Autograd Mechanism](https://pytorch.org/docs/stable/autograd.html#module-torch.autograd)
* [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
