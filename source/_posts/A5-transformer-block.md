---
title: A5-transformer-block
date: 2025-08-26 17:10:10
tags:
  - Transformer
categories:
  - Assignment
comments: false
mathjax: true
---

对于本次作业，我们将使用前几次作业实现的模块，将多个模块进行拼接，组成 **Transformer Block**，并最终连接 **Embedding** 和 **Transformer Block**，组成一个最基本的 **Transformer** 模型。本次作业将实现三个模块，分别是 **Decoder KVCache**，**Decoder Layer** 和 **Decoder Block**。

#### Task 1: Transformer Decoder KVCache

大多数当代基于 Transformer 的大语言模型（LLMs），如 Llama 和 ChatGLM，采用的是 **decoder-only** 架构，并以 **causal language modeling（CLM）** 目标进行预训练。这意味着在<u>推理阶段</u>，它们必须遵循自回归（auto-regressive）生成范式，逐 token 地进行生成。该过程可以自然地划分为两个阶段：

- **Prefilling 阶段**：LLM 被输入一个完整的未见过的查询序列（query sequence），这些 token 之间在此前并未进行 attention 计算。模型执行一次前向传播，返回下一个 token 的概率分布，接着我们可以从中生成第一个 token。
- **Decoding 阶段**：之后，为了生成后续的 token，每次会将新生成的 token 作为新的输入再次送入 LLM，此时它需要对当前 token 以及所有先前的 token（包括原始输入中的 token 和先前生成的 token）的 keys 和 values 进行 attention 计算。

因此，为了储存并避免在 **decoding 阶段** 中每个 Transformer decoder 层都对先前 token 的 key 和 value 进行重复计算（详见 Task2），我们可以从 **prefilling 阶段** 开始对这些 key 和 value 进行缓存。当新生成的 token 输入模型时，直接从缓存中检索并复用已有的 key-value 张量，并在序列维度上进行更新，为下一个 token 的生成做好准备。

为了更好地管理缓存中用于存储、读取和更新历史 key-value 张量的数据，我们设计了一个简单的模块，作为数据结构，命名为 **`TransformerDecoderKVCache`**。以下是该模块的 API 参考接口，供你实现时参考：

- `__init__(qkv_layout, num_layers=1)`：根据给定的 `qkv_layout`（用于推断 kv 的形状）和 `num_layers`（便于预先知道层数，你可以据此预分配一些内部数据结构）来初始化缓存。
- `has(layer_idx)`：检查缓存中是否存在指定层的 key-value 张量。
- `get(layer_idx)`：获取指定层的 key-value 张量。如果使用的是变长注意力（`varlen attention`）且 `qkv_layout=AttnQKVLayout.THD`，则额外返回 `cu_seqlens`；否则返回 `None` 作为占位。
- `set(layer_idx, k, v, cu_seqlens=None)`：为指定层设置 key-value 张量（若该层已存在，则覆盖），若使用变长注意力，则应传入  `cu_seqlens`。
- `append(layer_idx, k, v, cu_seqlens=None)`：沿序列维度更新指定层已有的缓存内容（若该层不存在，则行为应与 `set` 一致），若使用变长注意力，则应传入 `cu_seqlens`。
- `reset()`：清空缓存内容并重置为初始状态。

当然，上述数据结构只是一个简单且基础的实现（参考文献中可见 Hugging Face Transformers 的相关实现）。在实际应用中，还有许多更精细的设计方案，能够有效减少 KV 缓存的内存占用，并提升解码效率（具体可参考文献中的 vLLM 相关文档）。

##### TODO

你需要按照上述 API 参考在 `src/modeling/transformer.py` 中实现 `TransformerDecoderKVCache` 模块。该模块将在后续任务中作为辅助组件使用，用于仅在推理阶段以简洁的方式管理每个 Transformer decoder 层的 KV 缓存。

{% note warning %}

1.传入 `set` 和 `append` 方法的所有参数都保证与缓存中已有的数据保持一致。例如，当 `qkv_layout` 为 `AttnQKVLayout.THD` 时，将提供对应的 `cu_seqlens`；同时，传入张量的属性，如 `dtype`、`device` 以及由 `cu_seqlens` 推断出的内部 `batch_size`，也都会与缓存中已有的数据一致。但即便如此，出于错误处理和确保缓存正确性的考虑，你仍然应该检查参数的一致性。

2.本任务对时间和空间复杂度没有要求，因此你可以自由地设计数据结构，只需确保其逻辑正确且资源开销在可接受范围内。

{% endnote %}

#### Task2: Transformer Decoder Layer

基于 decoder-only 架构的 Transformer 大语言模型（LLM），可以形象地比喻为一个“巨型汉堡”：

- 最上层和最下层的面包片，分别对应模型中的 **输入嵌入层** 和 **输出嵌入层**，它们负责在 **token 空间** 与 **潜在的 hidden 表示空间** 之间进行转换。
- 中间层层叠叠的“牛肉饼”，则由一系列 **decoder layer** 构成。这些层通过 **self-attention 机制** 实现 token 之间的交互，通过 **MLP 机制** 实现每个 token 内部的线性与非线性变换。

因此，本任务的目标是像主厨一样精心打造这“牛肉饼”中的一片——即 `TransformerDecoderLayer` 模块。在后续任务中，我们将把多片这样的“牛肉饼”叠加在一起，并加上“面包层”，最终构建出这个“巨无霸汉堡”（详见 Task3）。

一个 Transformer decoder layer 由两个主要的子层组成：

- **Self-Attention 层**：给定输入 $X$ ，满足 $X \in \mathbb{R}^{batch\_size \times seq\_len \times hidden\_size}$ ，记作 `[b, s, h]`。如果同时提供了 `cu_seqlens`，则此处的 `batch_size` 被约定为 1，而真实的 `batch_size`（记作 `inner_batch_size`）需要通过 `cu_seqlens` 推断出来，因为内部各序列在 `seqlen` 维度上被拼接在了一起。
  $$
  \begin{aligned}
  R &= X \\
  \tilde{X} &= \mathrm{Norm}(X) \\
  Q, K, V &= \mathrm{split}(\tilde{X} \times W_{QKV}) \\
  \tilde{Q}, \tilde{K} &= \mathrm{RoPE}(Q), \mathrm{RoPE}(K) \\
  \tilde{O} &= \mathrm{SelfAttn}(\tilde{Q}, \tilde{K}, V) \\
  O &= \tilde{O} \times W_O + R
  \end{aligned}
  \tag {1}
  $$

- **MLP** 层：给定输入 $X$（即上述 **self-attention** 层的输出 $O$），该层对输入 $X$ 归一化后，再进行 MLP 变换，并通过残差连接，得到最终的输出 $O$。
  $$
  \begin{aligned}
  R &= X \\
  \tilde{X} &= \mathrm{Norm}(X) \\
  \tilde{O} &= \mathrm{MLP}(\tilde{X}) \\
  O &= \tilde{O} + R
  \end{aligned}
  \tag {2}
  $$

为了充分利用我们在之前作业中构建的模块，这里我们将 **Norm** 实现为 `GroupRMSNorm`，**RoPE** 实现为 `NTKAwareRoPE`， **SelfAttn** 实现为 `OfflineSlidingWindowAttn` 或 `OnlineSlidingWindowAttn`，**MLP** 实现为 `DenseMLPWithLoRA` 或 `SparseMLPWithLoRA`。由于 `NTKAwareRoPE` 和 `OnlineSlidingWindowAttn` 属于 <u>Bonus</u> 任务，我们会提供一个封装好的模块以便同学们使用，若自己实现了相应模块也可在代码中进行替换。

为了支持<u>推理阶段</u>的前向传播，`TransformerDecoderLayer` 模块的 forward 方法还支持一个可选的 `kv_cache` 参数，该参数由 `TransformerDecoderKVCache`（参见 Task1）实例化，负责管理所有 `decoder layer` 的 kv 缓存。你需要从中获取当前 `decoder layer` 对应的缓存，利用缓存中的 key-value 与当前的 key-value 一起对当前 query 进行注意力计算，并通过调用我们在 Task1 中实现的相应接口更新缓存。

在使用缓存的 key-value 对当前 query 进行注意力计算时，attention mask 也必须保持对齐。回顾 **A4** 的 **Task 2**，当 query 与 key-value 在序列维度上的长度不一致时，我们已经按照 Flash Attention 的设置，将掩码对齐到右下角（详见参考文献中的 Flash Attention 接口示例）。因此，在推理阶段的解码过程中，这一问题将由 attention 子模块**自动处理**。由于当前的 query 始终位于历史 key-value 之后（即位置索引最大），它对应 attention mask 矩阵的**最后一行**，从而确保其能够与所有缓存的 key-value 计算注意力。

对于自己实现 `NTKAwareRoPE` 的同学来说，另一个相关的问题是，当前 query 可能是单个 token，其位置索引不再是从 0 开始，而是一个较大的位置索引，因此需要正确地分配位置编码。为此，我们在 `NTKAwareRoPE` 模块的 forward 方法中引入了一个新的可选参数 `offset: int = 0`，方便你稍作修改，使其支持对输入张量的所有位置索引统一平移一个固定的偏移量，也就是将原始索引范围 `[0, seq_len - 1]` 转换为 `[offset, offset + seq_len - 1]`。当然，这个新功能在本任务中不作强制验证，所以你完全可以忽略它，继续使用你旧版的 `NTKAwareRoPE` 实现，或者采用其他方案准确处理位置索引问题。我们提供的 `NTKAwareRoPE` 模块已经实现了该功能，可以直接通过接口进行调用。

为了方便管理 `TransformerDecoderLayer` 模块及其子模块的初始化，我们在 `src/modeling/transformer.py` 中提供了一个通用的配置数据类 `TransformerConfig`。该配置类包含了初始化 `TransformerDecoderLayer` 模块所需的所有参数（除了 `layer_idx`，它是一个可选参数，取值范围为 `[0, config.num_layers]`，用于手动指定当前解码器层的索引 id），具体参数说明见**附表2**。

##### TODO

你需要实现 `src/modeling/transformer.py` 中的 `TransformerDecoderLayer` 模块。该模块接收输入 $X$，累计序列长度 `cu_seqlens`，以及一个 *Optional* 的 `kv_cache` 作为输入。模块内部依次经过 `self_attention` 层（`offline/online self-attention`）和 `MLP` 层（`dense/sparse MLP` ），过程中使用 `Group RMS` 、`Linear` 以及残差连接。最终返回一个与输入 $X$ 形状相同的输出张量 $O$，作为下一层 `decoder layer` 的输入。

{% note warning %}

1. 为了为每个 `Decoder Layer` 中的子模块或子操作分配唯一的随机数种子，我们通常会在 **TransformerConfig** 中提供的基础种子上添加一些偏移量，以生成实际使用的随机种子，具体偏移规则如**附表1**所示。
2. 我们保证传入 `Decoder Layer` 的输入在格式上与 `qkv_layout` 以及 `cu_seqlens`（包括保存在 `kv_cache` 中的内容）保持一致。但出于错误处理和验证正确性的考虑，检查各个参数的一致性是一个良好的编程习惯。
3. 我们保证仅在满足以下所有条件时才使用 `OnlineSlidingWindowAttn`：
   - 序列长度等于 `max_seq_len`；
   - `kv_cache` 为 `None`；
   - `qkv_layout` 为 `AttnQKVLayout.BSHD`，因此 `cu_seqlens` 为 `None`；
   - `qkv_pack_format` 为 `AttnQKVPackFormat.Q_K_V`。
4. 输入的属性（如 `dtype` 和 `device`）可能与模型参数的属性不同，因此你需要特别注意，确保输出的属性与输入保持一致。

{% endnote %}

#### Task3: Transformer Decoder Block

在 Task2 的基础之上，我们继续实现 `TransformerDecoderBlock` 模块。该模块由多个 `TransformerDecoderLayer` 层堆叠组成，外部包裹输入嵌入层和输出嵌入层，前者由 `ParallelVocabEmbedding` 模块实例化，后者由标准的 `nn.Linear` 模块实例化。

遵循 Llama 的设计（详见参考中的 *Llama Model Module* 部分），输入嵌入层 `VocabEmb` 接收一个形状为 `[batch_size, seq_len]` 的 token ID 序列张量 `I`，并将其从词汇空间映射到隐藏空间，生成初始张量 `X_ini`，其形状为 `[batch_size, seq_len, hidden_size]`。随后，初始隐藏张量 `X_ini` 会依次传入 `L` 个堆叠的解码器层 `DecoderLayers`，以获得最终隐藏张量 `X_fin`。

输出嵌入层 `LMHead` 则以通过 `FinalNorm`（一个 `GroupRMSNorm` 模块的实例）归一化后的最终隐藏张量 `X_fin~` 为输入，将其从隐藏空间映射回词汇空间，输出每个 token 对应的词汇预测 logits，结果为 `[batch_size, seq_len, vocab_size]` 的张量 `Logits`。

解码器块的整个前向传播过程可以形式化地表示为如下公式：
$$
\begin{align*}
X_{\text{ini}} &= \mathrm{VocabEmb}(I) \\
X_{\text{fin}} &= \mathrm{DecoderLayers}_{L}(X_{\text{ini}}) \\
\tilde{X}_{\text{ini}} &= \mathrm{FinalNorm}(X_{\text{fin}}) \\
\mathrm{Logits} &= \mathrm{LMHead}(\tilde{X}_{\text{ini}})
\end{align*}
\tag{3}
$$
除了上述结构以外，`decoder layer` 还需初始化并维护一个 `TransformerDecoderKVCache` 模块的实例，用于统一管理所有 `decoder layer` 的 KV 缓存。在推理阶段（即 `self.training` 为 `False` 时），该缓存会在 for-loop 中传递给每一层。

`TransformerDecoderBlock` 模块在初始化时接收一个 `TransformerConfig` 数据类的实例，作为全局配置参数，并在内部将该配置与各层的层索引一同传递给每个 `decoder layer`。关于 `TransformerConfig` 数据类中各项配置的详细说明，可参见 Task2 和**附表2**。

此外，你还需要实现若干简洁且实用的接口，用于访问 KV 缓存并统计模型参数情况：

- `get_kv_cache()`：返回当前的 KV 缓存对象。
- `set_kv_cache(kv_cache: TransformerDecoderKVCache)`：设置新的 KV 缓存对象。
- `reset_kv_cache()`：调用 KV 缓存对象的 `reset()` 方法，重置缓存。
- `num_parameters(learnable_only: bool = False, unit: str = "1")`：以指定的数量单位（可选单位包括 `"1"`、`"K"`、`"M"`、`"B"`）返回模型参数总数；若 `learnable_only` 设为 `True`，则仅统计可训练参数。
- `num_memory_footprint(unit: str = "B")`：以指定的字节单位（可选单位包括 `"B"`、`"KB"`、`"MB"`、`"GB"`）返回模型参数占用的内存大小。

##### TODO

你需要实现 `TransformerDecoderBlock` 模块，该模块以 token ID 张量 $I$ 作为输入，输出词汇预测张量 `Logits`。同时，它负责管理所有`decoder layer`的 KV 缓存，并提供若干便捷接口，供用户访问 KV 缓存及查询模型参数的相关统计信息。

{% note warning %}

1. `decoder layer`的随机数种子设置详见 Task2注意事项和**附表1**。
2. 输入的 token ID 的 `device` 可能与参数的 `device` 不同，确保输出 `logits` 的 `device` 与输入 token ID 保持一致。
3. 在 `TransformerConfig` 数据类中，有一个特殊的 `bool` 变量 `lm_head_tied`，表示 `lm_head` 层与词汇嵌入层是否共享参数，而非分开设置并初始化（详情见参考文献中的 HF PretrainedModel Tie Weights）。
4. 我们采用标准的 `nn.Linear` 层来实现 `lm_head`，其初始化方法与直接初始化 `nn.Parameter` 张量略有不同（详情见参考文献中的 Llama PretrainedModel Init Weights）。

{% endnote %}

------

#### 附表1

| Sub-module or Sub-Operation                   | Basic Random Seed             | Offset  |
| --------------------------------------------- | ----------------------------- | ------- |
| `qkv_proj` in the `i`-th decoder layer        | `config.proj_init_seed`       | `i + 1` |
| `o_proj` in the `i`-th decoder layer          | `config.proj_init_seed`       | `i + 2` |
| `attn_norm` in the `i`-th decoder layer       | `config.init_base_seed`       | `i + 1` |
| `attn` in the `i`-th decoder layer            | `config.init_base_seed`       | `i + 2` |
| `mlp_norm` in the `i`-th decoder layer        | `config.init_base_seed`       | `i + 3` |
| `mlp` in the `i`-th decoder layer             | `config.init_base_seed`       | `i + 4` |
| `softmax_dropout` in the `i`-th decoder layer | `config.softmax_dropout_seed` | `i`     |
| `lora` in the `i`-th decoder layer            | `config.lora_init_base_seed`  | `i`     |
| `lora_dropout` in the `i`-th decoder layer    | `config.lora_dropout_seed`    | `i`     |
| `vocab_embed` in the decoder block            | `config.init_base_seed`       | `0`     |
| `lm_head` in the decoder block                | `config.proj_init_seed`       | `0`     |

#### 附表2

| **Config Name**          | **Type**                 | **Default**               | **Required** | **Fixed** | **Description**                                              |
| ------------------------ | ------------------------ | ------------------------- | ------------ | --------- | ------------------------------------------------------------ |
| `num_layers`             | `int`                    | `None`                    | `True`       | `False`   | The number of transformer decoder layers used in `TransformerDecoderBlock`. |
| `hidden_size`            | `int`                    | `None`                    | `True`       | `False`   | The dimension of the hidden states.                          |
| `ffh_size`               | `int`                    | `None`                    | `True`       | `False`   | The dimmension of the intermediate hidden states used in `DenseMLPWithLoRA`and `SparseMLPWithLoRA`. |
| `max_seq_len`            | `int`                    | `None`                    | `True`       | `False`   | The maximum sequence length used in `NTKAwareRoPE` and `OnlineSlidingWindowAttn`. |
| `param_dtype`            | `torch.dtype`            | `torch.float32`           | `False`      | `False`   | The data type of **ALL** the parameters.                     |
| `param_device`           | `str`                    | `"cpu"`                   | `False`      | `False`   | The device on which **ALL** of the parameters are located.   |
| `init_base_seed`         | `int`                    | `42`                      | `False`      | `False`   | The basic random seed for parameter initialization.          |
| `rank`                   | `int`                    | `0`                       | `False`      | `True`    | The rank of the process, fixed to `0`.                       |
| `world_size`             | `int`                    | `1`                       | `False`      | `True`    | The number of processes, fixed to `1`.                       |
| `process_group`          | `Optional[ProcessGroup]` | `None`                    | `False`      | `True`    | The process group for distributed training, fixed to `None`. |
| `vocab_size`             | `int`                    | `None`                    | `True`       | `False`   | The size of the vocabulary used in `ParallelVocabEmbedding`and `lm_head` layer. |
| `vocab_init_mean`        | `float`                  | `0.0`                     | `False`      | `False`   | The mean value of the normal distribution to initialize the vocabulary embedding table in `ParallelVocabEmbedding`. |
| `vocab_init_std`         | `float`                  | `1.0`                     | `False`      | `False`   | The standard deviation of the normal distribution to initialize the vocabulary embedding table in `ParallelVocabEmbedding`. |
| `rope_base`              | `int`                    | `10000`                   | `False`      | `False`   | The base value to control the frequences in `NTKAwareRoPE`.  |
| `rope_ratio`             | `int`                    | `1`                       | `False`      | `False`   | The scaling ratio to extraplolate the frequencies used in `NTKAwareRoPE`. |
| `rope_dynamic`           | `bool`                   | `False`                   | `False`      | `False`   | Whether to dynamically update cached cos/sin embeddings in `NTKAwareRoPE`. |
| `group_size`             | `Optional[int]`          | `None`                    | `False`      | `False`   | The group size to split the hidden size in `GroupRMSNorm`.   |
| `eps`                    | `float`                  | `1e-5`                    | `False`      | `False`   | The epsilon value to avoid numerical instability in `GroupRMSNorm`. |
| `norm_init_range`        | `tuple`                  | `(-1.0, 1.0)`             | `False`      | `False`   | The range of the uniform distribution to initialize the scaling parameters in `GroupRMSNorm`. |
| `proj_init_seed`         | `int`                    | `42`                      | `False`      | `False`   | The random seed to initialize projection matrices, including `qkv_proj`, `o_proj`, as well as the `lm_head` if `lm_head_tied=False`. |
| `proj_init_mean`         | `float`                  | `0.0`                     | `False`      | `False`   | The mean value of the normal distribution to initialize projection matrices. |
| `proj_init_std`          | `float`                  | `1.0`                     | `False`      | `False`   | The standard deviation of the normal distribution to initialize projection matrices. |
| `lm_head_tied`           | `bool`                   | `False`                   | `False`      | `False`   | Whether to tie the weights of the `lm_head` layer to the one of the vocab embedding layer. |
| `online_attn_block_size` | `Optional[int]`          | `None`                    | `False`      | `False`   | The block size for `OnlineSlidingWindowAttn`. If `None`, use `OfflineSlidingWindowAttn`instead. |
| `head_dim`               | `int`                    | `None`                    | `True`       | `False`   | The dimension of each attention head.                        |
| `num_q_head`             | `int`                    | `None`                    | `True`       | `False`   | The number of query heads.                                   |
| `num_kv_head`            | `int`                    | `None`                    | `True`       | `False`   | The number of key/value heads.                               |
| `qkv_pack_format`        | `AttnQKVPackFormat`      | `AttnQKVPackFormat.Q_K_V` | `False`      | `False`   | The packing format for QKV tensors.                          |
| `qkv_layout`             | `AttnQKVLayout`          | `AttnQKVLayout.BSHD`      | `False`      | `False`   | The shape layout for QKV tensors.                            |
| `window_size`            | `Optional[int]`          | `None`                    | `False`      | `False`   | The window size for sliding window attention.                |
| `causal`                 | `bool`                   | `False`                   | `False`      | `False`   | Whether to apply causal mask to the attention.               |
| `softmax_dropout_rate`   | `float`                  | `0.0`                     | `False`      | `False`   | The dropout rate applied after the softmax operation.        |
| `softmax_dropout_seed`   | `int`                    | `42`                      | `False`      | `False`   | The random seed for softmax dropout.                         |
| `softmax_scale`          | `Optional[float]`        | `None`                    | `False`      | `False`   | The scaling factor applied to the softmax logits.            |
| `softmax_cap`            | `Optional[float]`        | `None`                    | `False`      | `False`   | The capping value to apply `softmax capping` to adaptively control the magnitude of the softmax logits, if `None`, use `softmax temperature` trick instead. |
| `softmax_temp`           | `float`                  | `1.0`                     | `False`      | `False`   | The temperature value to apply `softmax temperature` to control the sharpness of the softmax distribution when `softmax capping` is disabled. |
| `softmax_clip_range`     | `Tuple[float, float]`    | `(0.0, 1.0)`              | `False`      | `False`   | The clipping range to apply `softmax clipping` to prevent the outliers in the softmax weights. |
| `apply_qk_norm`          | `bool`                   | `False`                   | `False`      | `False`   | Whether to apply `QK layer normalization` to the query and key tensors. |
| `qk_norm_group_size`     | `Optional[int]`          | `None`                    | `False`      | `False`   | The specific group size for `QK layer normalization` if enabled. Other configurations for `QK layer normalization` share the same as above. |
| `activation_type`        | `MLPActivationType`      | `MLPActivationType.SILU`  | `False`      | `False`   | The activation function type used in the mlp layer.          |
| `lora_rank`              | `int`                    | `0`                       | `False`      | `False`   | The rank for LoRA.                                           |
| `lora_alpha`             | `Optional[float]`        | `None`                    | `False`      | `False`   | The alpha parameter for LoRA.                                |
| `lora_dropout_rate`      | `float`                  | `0.0`                     | `False`      | `False`   | The dropout rate for LoRA layers.                            |
| `lora_dropout_seed`      | `int`                    | `42`                      | `False`      | `False`   | The random seed for LoRA dropout.                            |
| `lora_init_base_seed`    | `int`                    | `42`                      | `False`      | `False`   | The base random seed to initialize the parameters of LoRA.   |
| `num_experts`            | `Optional[int]`          | `None`                    | `False`      | `False`   | The number of experts for `SparseMLPWithLoRA`. If `None`, then use `DenseMLPWithLoRA` instead. |
| `moe_topk`               | `int`                    | `1`                       | `False`      | `False`   | The top-k value for expert routing in `SparseMLPWithLoRA`.   |
| `gate_init_mean`         | `float`                  | `0.0`                     | `False`      | `False`   | The mean value of the normal distribution to initialize the gating parameters. |
| `gate_init_std`          | `float`                  | `1.0`                     | `False`      | `False`   | The standard deviation of the normal distribution to initialize the gating parameters. |

#### References

*提示：以下是一些可能对你的任务有帮助的参考资料，或者可以加深/拓展你对 Transformer Block 的理解：*

**!! 请记住：查阅论文、源码以及官方文档，并从中进行思考和学习，是一项基本且至关重要的能力。请尽量不要过度依赖一些带有偏见或内容浅显的博客，例如 CSDN !!**

* [HF Transformers Dynamic Cache Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/cache_utils.py#L351)
* [vLLM Paged Attention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html)
* [vLLM Chunked Prefill](https://docs.vllm.ai/en/latest/models/performance.html)
* [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html)
* [vLLM Fp8 E4M3 KV Cache](https://docs.vllm.ai/en/latest/quantization/fp8_e4m3_kvcache.html)

* [Llama MLP Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L229)
* [Llama Attention Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L275)
* [Llama DecoderLayer Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L626)
* [Flash Attention Interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)

* [Llama Model Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L830)
* [Llama PretrainedModel Init Weights](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L739)
* [HF PretraiedModel Tie Weights](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/modeling_utils.py#L1915)