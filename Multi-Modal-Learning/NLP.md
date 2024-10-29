
[一文彻底搞懂 Bert](https://zhuanlan.zhihu.com/p/694502940)
-  Pre-training： 存在通用的语言模型，先用文章预训练通用模型，然后再根据具体应用，用 supervised 训练数据，精加工（fine tuning）模型，使之适用于具体应用。
-  Deep Bidirectional Transformers 模型
-  论文的主要贡献
   - 证明了双向预训练对语言表示的重要性。与之前使用的单向语言模型进行预训练不同，BERT使用遮蔽语言模型来实现预训练的深度双向表示。
   - 预先训练的表示免去了许多工程任务需要针对特定任务修改体系架构的需求。
- 模型
   - multi-layer bidirectional Transformer编码器

[如何最简单、通俗地理解Transformer？](https://www.zhihu.com/question/445556653/answer/3254012065)
- Transformer引入的自注意力机制能够有效捕捉序列信息中长距离依赖关系，相比于以往的RNNs，它在处理长序列时的表现更好。
- 自注意力机制的另一个特点时允许模型并行计算，无需RNN一样t步骤的计算必须依赖t-1步骤的结果，因此Transformer结构让模型的计算效率更高，加速训练和推理速度。
- 基本概念[Transformer模型详解](https://zhuanlan.zhihu.com/p/338817680)
  - Transformer 中单词的输入表示 x由单词 Embedding 和位置 Embedding （Positional Encoding）相加得到。
    - Transformer 的内部结构图，左侧为 Encoder block，右侧为 Decoder block。红色圈中的部分为 Multi-Head Attention，是由多个 Self-Attention组成的，可以看到 Encoder block 包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention (其中有一个用到 Masked)。Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。
    -  Self-Attention 的结构，在计算的时候需要用到矩阵Q(查询),K(键值),V(值)。在实际中，Self-Attention 接收的是输入(单词的表示向量x组成的矩阵X) 或者上一个 Encoder block 的输出。而Q,K,V正是通过 Self-Attention 的输入进行线性变换得到的。
  - Query：Query（查询）是一个特征向量，描述我们在序列中寻找什么，即我们可能想要注意什么。
  - Keys：每个输入元素有一个键，它也是一个特征向量。
  - Values：每个输入元素，我们还有一个值向量。
  - Score function：评分函数，为了对想要关注的元素进行评分，我们需要指定一个评分函数f该函数将查询和键作为输入，并输出查询-键对的得分/注意力权重。它通常通过简单的相似性度量来实现，例如点积或MLP。
- Encoder结构

[史上最全Transformer面试题系列（一）](https://zhuanlan.zhihu.com/p/148656446)
- Transformer为何使用多头注意力机制？
  - 每个注意力头可以被看作一个独立的视角。通过设置多个注意力头，Transformer 能够在不同的维度上理解数据中的关系。例如，一个注意力头可能关注句子中的短程依赖关系（如词组之间的联系），而另一个头可能关注长程依赖关系（如句首和句尾的关系）。这种多视角的关注使模型能够更全面地捕捉序列中的不同特征。
  - 每个注意力头通过独立的查询Q、键K 和值V 向量来计算注意力分布。多头注意力机制将这些头的结果进行拼接后再投影回输出空间。这样做可以让模型从多个嵌入空间中抽取信息，显著增加了模型的表达能力。
- Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？
    - 区分不同的作用： 查询向量（Q）表示我们希望关注的内容，而键向量（K）表示序列中每个位置的信息特征。
- Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？
  - 点乘能够直接测量两个向量之间的相似度。对于两个向量 Q 和 K 的点乘结果，当它们越接近时，点积值越大，这正是我们在注意力机制中想要捕捉的特性，即：查询与键越相关，其注意力权重越高。
- 在计算attention score的时候如何对padding做mask操作？
  - 在计算注意力分数（attention score）时，padding mask 用于忽略填充（padding）位置的值。填充位置通常是在对不等长序列进行批量处理时引入的，用于对齐序列长度。
  - Padding Mask 通过以下步骤处理填充位置：
    - 生成一个布尔型的填充 mask 矩阵，将填充位置标记为 True。
    - 扩展 mask 的维度，使其可以与注意力分数矩阵形状匹配。
    - 将填充位置的注意力分数设置为 −∞，通过 softmax 确保填充位置的权重为 0。
- 为什么在进行多头注意力的时候需要对每个head进行降维？
  - 控制计算复杂度
  - 保持整体输出维度不变
  - 避免过度拟合
- 大概讲一下Transformer的Encoder模块？
  - 编码器模块包含多个相同结构的编码器层，每层包括两个主要部分：
    - 多头自注意力机制（Multi-Head Self-Attention）
    - 前馈神经网络（Feedforward Neural Network, FFN）
    - 残差连接（Residual Connection）和层归一化（Layer Normalization）
- 简单介绍一下Transformer的位置编码？有什么意义和优缺点？
  - 由于 Transformer 没有循环结构或卷积结构，缺少自然的顺序信息，因此位置编码通过给每个位置添加不同的编码值，使模型能够识别序列中元素的顺序。
  - 包括可学习的位置编码、相对位置编码、旋转位置编码和一些应用于图像或长序列的改进编码方法。
  - Transformer 中的残差结构（Residual Connection）是一种将每个子层的输入直接加到输出上的连接方式。每个子层后都使用残差连接，并结合层归一化（Layer Normalization），它帮助模型更好地训练并提高计算稳定性。
    - 缓解梯度消失问题
    - 增强信息流动
    - 加速收敛
    - 避免过拟合和增强模型鲁棒性
- 为什么 Transformer 使用 LayerNorm 而不是 BatchNorm？
  - 适合变长序列处理
  - 适用于自注意力机制
  - 更稳定的训练和泛化能力
- Encoder 和 Decoder 的交互过程
  - Encoder 生成上下文表示
    - 在 Transformer 的 Encoder 端，输入序列（如句子的词）会通过多个编码器层处理，每层包含多头自注意力机制和前馈网络，最后生成每个输入位置的上下文表示。
    - Encoder 的最终输出包含了序列中每个位置的全局信息（即每个词及其与其他词的关系），构成了一组上下文表示向量，作为 Decoder 的输入。
  - Decoder 利用 Encoder-Decoder Attention 获取 Encoder 信息

[NLP关键词提取方法总结及实现](https://blog.csdn.net/asialee_bird/article/details/96454544)
- 基于统计特征的关键词提取（TF-IDF）
- 基于词图模型的关键词提取（TextRank）
- 基于主题模型的关键词提取（LDA）
