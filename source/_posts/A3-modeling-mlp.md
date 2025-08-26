---
title: A3 Modeling MLP
date: 2025-06-29 21:54:26
tags:
  - MLP
  - LoRA
categories:
  - Assignment
comments: false
mathjax: true
---

对于本次作业，我们将继续 Modeling 任务，以帮助你更深入地理解 Transformer 的各个组成模块。本次将特别关注 Transformer 结构核心的关键层之一：**MLP** 层。

# Task 1: Dense MLP with LoRA Adapters

## Part 1: Dense MLP

Multi-Layer Perceptron (MLP) 模块是深度学习中的一个基本模块，特别适用于处理复杂模式和非线性关系的任务。它已被广泛应用于基于 Transformer 的 LLMs 中，作为与 Attention 模块并列的核心组件。当前主流 LLM（如 Llama）中使用的 MLP 模块，基本上遵循 Gated Linear Units (GLU) 的结构风格（具体细节可参考 GLU 论文），具体形式如下：

$$
\text{MLP}(\mathbf{X}) = (\phi(\mathbf{X} \times \mathbf{W}_{gate}) \odot (\mathbf{X} \times \mathbf{W}_{up})) \times \mathbf{W}_{down}
$$

其中：
- $\mathbf{X}$ 表示输入 `hidden states`，满足 $\mathbf{X} \in \mathbb{R}^{\text{batch\_size} \times \text{seq\_len} \times \text{hidden\_size}}$，记为 `[b, s, d]`。
- $\mathbf{W}_{\text{up}}$ 表示上投影矩阵，满足 $\mathbf{W}_{\text{up}} \in \mathbb{R}^{\text{hidden\_size} \times \text{ffh}}$，用于将 $\mathbf{X}$ 从 `h` 维映射到 `ffh` 维，记为 `[d, ffh]`。
- $\mathbf{W}_{\text{down}}$ 表示下投影矩阵，满足 $\mathbf{W}_{\text{down}} \in \mathbb{R}^{\text{ffh} \times \text{hidden\_size}}$，用于将 $\mathbf{X}$ 从 `ffh` 维映射回 `h` 维，记为 `[ffh, d]`。
- $\mathbf{W}_{\text{gate}}$ 表示门控投影矩阵，满足 $\mathbf{W}_{\text{gate}} \in \mathbb{R}^{\text{hidden\_size} \times \text{ffh}}$，类似传统的深度 RNN 架构，GLU 为了引入非线形变换而引入 $\mathbf{W}_{\text{gate}}$，配合激活函数 $\phi(·)$ 来形成门控项 $\phi(\mathbf{X} \times \mathbf{W}_{gate})$，其中 $\odot$ 表示逐元素乘（element-wise product），以此在前向传播中控制信息流动，在反向传播中缓解梯度消失问题。


### TODO

**完成 `src/modeling/mlp.py` 中的 `DenseMLPWithLoRA` 模块**，实现上述定义的 GLU-style MLP 模块，具体细节包括：
- 对于 `DenseMLPWithLoRA` 模块，激活函数 $\phi(·)$ 是可配置的，通过传入名为 `activation_type` 的参数进行指定，该参数作为 `src/modeling/mlp.py` 中已定义枚举类 `MLPActivationType` 的一个实例。`activation_type` 与激活函数之间的映射关系，参考了参考文献中提供的 GLU Variants 论文以及部分 PyTorch 的实现方式。
- 同样，作为一个可学习的模块，你需要为 `DenseMLPWithLoRA` 模块实现 `reset_parameters` 方法，用于对三个投影矩阵进行初始化，初始化方式为从**正态分布**中采样。但与之前的 `Norm` 层和 `Embedding` 层不同，对于投影矩阵的初始化，我们通常采用 `Xavier Initialization` 或 `Kaiming Initialization`（具体细节可参考参考文献）:
  - 如果 `activation_type` 使用 `MLPActivationType.SIGMOID` 或 `MLPActivationType.BILINEAR`，则使用 `Xavier Initialization`；
  - 否则，对于其余 `ReLU-family` 的激活函数，则使用 `Kaiming Initialization`；
  - 注意，当指定 `ReLU-family` 的激活函数时，要求使用 `Kaiming Initialization` 方法（uniform 或 norm），此时，**我们约定使用 `fan_in mode`**，即 `Kaiming Initialization` 中标准差 std 应该由 $\sqrt \frac{2}{\text{fan\_in}}$ 得到。但实际上，Pytorch 的 `nn.Linear` 模块初始化的参数矩阵其 `shape` 是 $\text{[out\_features, in\_features]}$，而 Pytorch 的 `Kaiming Initialization` 方法也保持了这个习惯。所以当你在实现 `reset_parameters` 时请特别注意这一点。
- 同样，我们提供一个基随机数种子 `init_base_seed`，为了避免不同投影矩阵具有相同的初始化结果，你需要为每个投影矩阵分配一个唯一的 seed 偏移量，具体而言：
  - $\mathbf{W}_{\text{up}}$ 对应 `seed = init_base_seed + 1`；
  - $\mathbf{W}_{\text{gate}}$ 对应 `seed = init_base_seed + 2`；
  - $\mathbf{W}_{\text{down}}$ 对应 `seed = init_base_seed + 3`。

{% note warning %}
1. 我们省略了所有线性投影中的偏置项（bias）。
2. 参数中的 `dtype` 和 `device` 是针对可学习参数的设置，可能与输入 $\mathbf{X}$ 的 `dtype` 和 `device` 不同，即不强制匹配，允许你控制这些行为，这样设计更灵活。
3. 输出 O 的属性（包括 `dtype` 和 `device`）必须与输入 $\mathbf{X}$ 保持一致。
4. `reset_parameters` 方法应该在 `__init__` 方法中自动调用一次，用于初始化所有参数。
{% endnote %}


## Part 2: Dense MLP with LoRA Adapters

实际上，在大模型微调中，由于 MLP 模块中的参数通常占据了 LLMs 中超过 90% 的可训练参数，因此采用全量线性参数监督微调（supervised fine-tuning, SFT）的方式效率非常低，特别是在线性参数带来的增益（记作 $\Delta(\mathbf{W})$）高度稀疏的情况下。为了实现关于 LLMs 的参数高效微调（parameter-efficient fine-tuning, PEFT），提出了一种称为 Low-Rank Adaptation（LoRA） 的方法（具体细节见参考文献），该方法现已成为 PEFT 中最流行的策略之一。

LoRA 的核心基本假设是：**$\Delta(\mathbf{W})$ 是一个低秩且稀疏的矩阵**，其可以通过低秩分解表示为：

$$
\Delta(\mathbf{W}) = \frac{\alpha}{r} \mathbf{A}_{\text{r}} \times \mathbf{B}_\text{r}
$$

其中：
- $\mathbf{A}_\text{r} \in \mathbb{R}^{\text{h} \times \text{r}}$，$\mathbf{B}_\text{r} \in \mathbb{R}^{\text{r} \times \text{h}}$ 是一对低秩分解的投影矩阵；
- $\alpha$ 是一个可配置的缩放因子，用于控制 $\Delta(\mathbf{W})$ 的数值大小。

基于以上内容，带 LoRA adapters 的 MLP 模块的计算可以分解为如下公式，其中引入了标准的 `Dropout` 层（dropout rate 为 $p$）来进一步增强 $\Delta(\mathbf{W})$ 的稀疏性：

$$
\text{MLP}_{\text{LoRA}}(\mathbf{X}) = \text{MLP}(\mathbf{X}) + \text{Dropout}_p(\mathbf{X} \times \Delta(\mathbf{W})) = \text{MLP}(\mathbf{X}) + \text{Dropout}_p(\frac{\alpha}{r}\mathbf{X} \times \mathbf{A}_{\text{r}} \times \mathbf{B}_\text{r})
$$

通过这种方式，我们只需要在监督微调（SFT）过程中训练可学习的 $\mathbf{A}_r$ 和 $\mathbf{B}_r$，而冻结所有其他预训练的投影矩阵，从而实现参数高效的微调策略。

### TODO

在满足 Task1 的要求基础上，**进一步完善 `src/modeling/mlp.py` 中的 `DenseMLPWithLoRA` 模块**，实现上述定义的 GLU-style MLP with LoRA Adapters 模块，具体细节包括：
- 无论是使用 `Xavier Initialization` 还是 `Kaiming Initialization`，$\mathbf{A}_{\text{r}}$ 和 $\mathbf{B}_\text{r}$ 都应从 **uniform 分布**中进行初始化。
- 我们额外提供了一个 `lora_init_base_seed` 参数，用于控制 LoRA 相关参数矩阵的初始化，具体而言：
  - $\mathbf{A}_{\text{r}}$ 对应 `seed = lora_init_base_seed + 1`；
  - $\mathbf{B}_{\text{r}}$ 对应 `seed = lora_init_base_seed + 2`。
- 为了确保前向传播的可复现性，我们额外提供了一个 `lora_dropout_seed` 和 `lora_dropout_rate` 参数，用于控制 `Dropout` 层的随机行为。


{% note warning %}
1. 本任务中对 LoRA 的用法进行了简化，即整个 MLP 模块中只应用一次 LoRA。但在实际工程中，更推荐为 MLP 模块中的每个线性投影矩阵分别设计一个 LoRA 适配器。
2. 参数 `lora_rank` 保证在有效范围 $[0, min(h, ffh)]$ 之内，若 `lora_rank = 0`，你应跳过任何与 LoRA 有关的逻辑，即 LoRA 是 `DenseMLPWithLoRA` 中的可选模块。
3. 参数 `lora_alpha` 是一个正缩放因子，默认为 `None` 时，表示应将其设置为与 `lora_rank` 相同的值。
4. 你当然可以参考 LLaMA、ChatGLM、PEFT 等开源项目中如何实现（带 LoRA ）MLP 模块，具体内容可见参考文献。但要注意：本任务中所列的具体要求与这些项目实现略有不同，请严格按照当前任务说明进行设计与实现。
{% endnote %}

# Dense MLP 小结

综上，你需要实现 `DenseMLPWithLoRA` 模块，其功能包括：
1. 初始化所有可学习参数矩阵，其中包括：
    - 基本的投影矩阵（由 `init_base_seed` 控制）；
    - LoRA adapters 的相关参数矩阵（由 `lora_rank`、`lora_alpha`、`lora_dropout_rate`、`lora_dropout_seed` 和 `lora_init_base_seed` 控制）；
2. 接收输入 X，执行 GLU-style MLP with LoRA adapters 的 forward 计算过程，其中激活函数由 `activation_type` 指定；
3. 最后输出的 `hidden_states`，记为 O 应与 X 具有相同形状。


# Task 2: Sparse MLP

在 Task 1，2 中实现的 `DenseMLPWithLoRA` 模块的基础上，我们将继续结合主流 Mixture-of-Experts (MoE) 架构实现 `SparseMLPWithLoRA` 模块（更多细节见参考文献）。首先，所谓 **Dense** 的 MLP 模块，通常是指一种标准结构：它先将 `hidden_states` $\mathbf{X}$ 从 `h` 维上投影（up-project）到更高的 `ffh` 维，再通过 `gating` 机制下投影（down-project）回原始维度。

对于 **Sparse** 的 MLP 模块，类似于 attention 模块中的 multi-head 机制，将投影矩阵的 `ffh` 维度划分为 `ne` 个大小相等的 `shard`（分片），每个 `shard` 的大小为 `e = ffh // ne`，对应一个“专家”（expert）$E_i$，其中 $i \in [0, …, \text{ne}−1]$（`ne` 表示专家的数量）。因此，与传统使用大维度 `ffh` 的 Dense 投影不同，`SparseMLPWithLoRA` 模块中，`hidden_states` $\mathbf{X}$ 中的每个 `token` 仅通过一个 **routing mechanism** 映射到 `k` 个特定的 experts，每个 expert 仅负责处理一个特定的 `e` 维子空间（在本模块中，你可以简单地**将每个 expert 建模为一个小型的 DenseMLPWithLoRA 模块**，其中 `ffh_size` 参数设置为 `e`）。最终，每个 `token` 的最终输出是来自这 `k` 个 experts 子输出的加权和。通过这种方式，我们可以同时实现两个目标：
- 降低高维计算开销；
- 提高潜在模式的多样性，其增益比例约为 `ne`，类似于 multi-head 机制带来的并行子空间学习能力。

具体而言，有两个问题需要考虑：
1. 如何建模 **routing mechanism**，为每个 `token` 选择 `k` 个特定的 experts？
2. 如何确定权重 $\mathbf{W}$，为每个 `token` 对应 `k` 个 experts 得到的子输出进行加权求和？

对于上述两个问题，存在多种解决方案，而在本任务中，我们选择参考 Mixtral 的方法（具体细节见参考文献），采用如下方案：

1. 如下方公式所示，我们引入一个额外的线性 `gating` 层 $\mathbf{G}$，满足 $\mathbf{G} \in \mathbb{R}^{\text{h} \times \text{ne}}$，对于每个 `token t`，满足 $\text{t} \in \mathbb{R}^{\text{b} \times 1 \times \text{h}}，$通过 $\mathbf{G}$ 将其 `hidden_states` 投影为一个 `ne` 维的 `logits`，然后对该 `logits` 应用 `softmax`，形成一个 `ne` 维的 routing 概率分布 $\mathbf{P}_\text{t}$，其中 $\mathbf{P}_\text{t}[\text{i}]$ 表示该 `token` 被路由到 expert $E_\text{i}$ 的概率：

  $$
  \mathbf{P}_\text{t} = \text{softmax}(\mathbf{X}_\text{t} \times \mathbf{G}), \quad where\space \mathbf{G} \in \mathbb{R}^{\text{h} \times \text{ne}}, \space\forall \text{t}
  $$

2. 基于 $\mathbf{P}_\text{t}$，我们从中**选出概率最高的 `k` 个 experts** 组成一个集合 $\mathbf{I}_\text{t}$，作为该 `token` 的路由，这 `k` 个概率值另外构成一个新的 `k` 维**未归一化分布 $\mathbf{Q}_\text{t}$**：

  $$
  \mathbf{I}_\text{t} = \text{arg-topk}(\mathbf{P}_\text{t}), \quad \mathbf{Q}_\text{t} = \mathbf{P}_\text{t}[\mathbf{I}_\text{t}]
  $$

3. 我们重新对 $\mathbf{Q}_\text{t}$ 进行归一化（renormalization），定义新的 `k` 维 routing 的概率分布，从而得到关于该 `token` 每个 expert 输出的加权权重 $\mathbf{W}_\text{t}$ 以及每个 expert $E_\text{i}$ 的输出：

  $$
  \mathbf{W}_\text{t} = \frac{\mathbf{Q}_\text{t}}{\text{sum}(\mathbf{Q}_\text{t})}, \space\forall \text{t}
  $$

  $$
  \mathbf{O}_\text{t}'[\text{i}] = E_\text{i}(\mathbf{X}_\text{t}), \space\forall \text{i} \in \mathbf{I}_\text{t}, \space\forall \text{t}
  $$

4. 此外，为了模拟类似于 A2 `ParallelVocabEmbedding` 中的分布式环境，我们为 `SparseMLPWithLoRA` 模块添加了两个类似的参数：`rank` 和 `world_size`，其含义同 A2。这意味着你应该仅为当前模块实例化 `nle` 个本地 experts，其中 expert 的索引范围为 $R = [rank * nle, (rank + 1) * nle)$，其中，`nle = ne // world_size` 表示每个进程（rank）中包含的本地 experts 数量。因此，对于每个 `token t`，`SparseMLPWithLoRA` 模块的最终输出 $\mathbf{O}_\text{t}$ 只是一个**部分加和**（partial sum），即只计算该 `token` 被路由到的 expert 子集 $\mathbf{I}_\text{t}$ 与本地 expert 索引集合的交集（记作 $\mathbf{I}_\text{t}’$），如果 $\mathbf{I}_\text{t}’$ 为空，则输出应得到**全零向量**。理论上，最终的完整输出应该通过对所有 `rank` 聚合得到，但我们省略该过程，即只计算本地输出即可：

  $$
  \mathbf{O}_\text{t} = \begin{cases}
    \sum\limits_{\text{i} \in \mathbf{I}_\text{t}’} \mathbf{W}_\text{t}[\text{i}] \mathbf{O}_\text{t}'[\text{i}], & \mathbf{I}_\text{t}’ \neq \emptyset \\
    \vec{\mathbf{0}}, & \mathbf{I}_\text{t}’ = \emptyset
    \end{cases}, \quad where\space \mathbf{I}_\text{t}’ = \mathbf{I}_\text{t} \space\cap\space R, \space\forall t
  $$

## TODO

**完成 `src/modeling/mlp.py` 中的 `SparseMLPWithLoRA` 模块**，具体细节包括：
- 同样，你需要为 `SparseMLPWithLoRA` 模块实现 `reset_parameters` 方法：
  - 对于 experts，即 `DenseMLPWithLoRA`，你可以直接调用它们各自的 `reset_parameters` 方法，注意，为了避免相同的初始化结果，你应该为每个 expert 的 seed 设置一个偏移，具体而言，`DenseMLPWithLoRA` 的 `init_base_seed`，`lora_dropout_seed`，`lora_init_base_seed` 偏移量定义为**该 expert 的全局索引 index**。
  - 对于 `gating` 层 $\mathbf{G}$，直接通过 `nn.init.normal_` 初始化，均值和标准差分别由参数 `init_mean` 和 `init_std` 控制，其随机性由 `init_base_seed` 控制，**且不加任何偏移量**。


{% note warning %}
1. 我们继承 Task1，Task2 中关于 `DenseMLPWithLoRA` 子模块的注意事项，用于建模本地 expert。
2. 参数中的 `dtype` 和 `device` 是针对可学习参数的设置，可能与输入 $\mathbf{X}$ 的 `dtype` 和 `device` 不同，即不强制匹配，允许你控制这些行为，这样设计更灵活。
3. 输出 O 的属性（包括 `dtype` 和 `device`）必须与输入 $\mathbf{X}$ 保持一致。
4. `reset_parameters` 方法应在 `__init__` 方法中自动调用一次，用于初始化所有参数。
5. `Gating` 层 $\mathbf{G}$ 的权重通常需要更高的精度，因为后续的 `softmax` 操作对数值较为敏感。因此，**$\mathbf{G}$ 的参数 `dtype` 固定为 float32**，不受传入的 `dtype` 参数影响。
6. `ffh` 保证能被 `ne` 整除，但在 `__init__` 方法中检查可整除性仍然是一个良好的编程习惯。
{% endnote %}

# Sparse MLP 小结

综上，你需要实现 `SparseMLPWithLoRA` 模块，其功能包括：
1. 初始化所有可学习参数，包括本模块负责的每个本地 expert 的参数，以及 `gating` 层 $\mathbf{G}$ 的参数。
2. 接收输入 $\mathbf{X}$，对于每个 `token t`，计算其 top-k expert 子集，仅对与当前 `rank` 管理的本地 expert 集合 R 的交集执行 `forward` 计算流程，对于未路由到本地 experts 的 `token`，其输出保持为全零向量。
3.	最终返回与输入 $\mathbf{X}$ 具有相同形状的输出 `hidden states` $\mathbf{O}$，最终某个 `token t` 的非零输出为路由到的本地 experts 所产生的子输出的加权和。

# References

* [Llama MLP Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L229)
* [ChatGLM MLP Module](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L459)
* [GLU Paper](https://arxiv.org/abs/1612.08083)
* [GLU Variants Paper](https://arxiv.org/abs/2002.05202)
* [PEFT Documentation](https://huggingface.co/docs/peft/index)
* [LoRA Paper](https://arxiv.org/abs/2106.09685)
* [PEFT LoRA-Linear Layer Implementation](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L400)
* [Pytorch SiLU Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html)
* [Pytorch GELU Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html)
* [Pytorch ReLU Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html)
* [Pytorch Sigmoid Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html)
* [Pytorch Kaiming Normal Initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_)
* [Pytorch Xavier Normal Initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_)
* [MoE Paper](https://arxiv.org/abs/1701.06538)
* [Mixtral Paper](https://arxiv.org/abs/2401.04088)
* [Mixtral MoE MLP Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/mixtral/modeling_mixtral.py#L610)

以上是一些可能对你完成任务有帮助的参考资料，也可以用来加深或拓宽你对 `Dense MLP` 层、`LoRA Adapter`、`稀疏 MoE（Mixture of Experts）MLP` 以及深度学习中激活函数的理解。


！！请记住：查阅论文、源码以及官方文档，并从中进行思考和学习，是一项基本且至关重要的能力。请尽量不要过度依赖一些带有偏见或内容浅显的博客，例如 CSDN！！
