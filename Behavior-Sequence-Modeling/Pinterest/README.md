# 22’ PinnerFormer

Sequence Modeling for User Representation @ Pinterest

![image.png](attachment:7c761ecb-6cc8-443d-9bf3-86fef12019f3:image.png)

https://newsletter.theaiedge.io/p/the-aiedge-how-pinterest-uses-transformers

### 1. Motivation

Pinterest 每月有超 4 亿活跃用户浏览数十亿 Pins（includes image, text, web links, and a board）。用户主要通过三类渠道发现灵感：

- 个性化 Homefeed
- 与当前 Pin 相关的 Related Pins
- 文本搜索Text Query

用户的互动（保存、点击、放大、隐藏等）为推荐系统提供反馈，因此需要精准刻画用户兴趣。

Pinterest 提出并部署了序列用户表征模型 **PinnerFormer**。利用用户历史 Pin 行为序列，通过”**dense all action loss**”训练，在离线批处理环境中生成长期兴趣embedding。

解决了实时序列模型的 *a)高计算开销 cost of computation* 与 *b)基建复杂度 infrastructure complexity*。

与其为每个模型分别生成一套user embedding, PinnerFormer 作为**通用**用户向量，可同时服务数十个下游排序模型。

### 2. Design Choices

**2.1 Single vs. multiple embeddings for a single user**

早期模型如**PinnerSage**可为每个用户生成可变数量的兴趣向量，但在排序模型中带来问题：

- 存储膨胀
- 计算开销
- 数据加载慢

因此PinnerFormer 改为**single embedding，** 且在离线评估中比 PinnerSage 更能反映用户长期兴趣。

**2.2 Real-time vs. Offline inference**

序列模型强调实时更新，但需付出：

- **高计算成本**：每次用户操作都要拉取完整历史并在复杂模型上推理
- **高基础设施复杂度**：需维护流式基础设施，处理数据损坏与恢复
    - 增量更新user embedding需要一套容错、回放、快照和快速预热的流式基础设施。

PinnerFormer 通过新的损失函数(How?)缩小了“实时模型”与“日更推理模型”之间的性能差距。

### 3. PinnerFormer

**数据/Data**

- **Pins**：规模达数十亿，每个 Pin 由 PinSage 生成 256 维向量，融合视觉、文本与交互信息。
- **用户**：5 亿 + 月活。每位用户记录其按时间排序的历史 Pin 交互序列，动作包括保存、点击、放大查看、评论等。为控制计算量，仅取最近 M 条行为。
- **Action Sequence：**使用对应 Pin 的 PinSage embedding并附带action metadata。

**学习目标/Learning Goals**

- 构建映射 f: 用户 → ℝᵈ 与 g: Pin → ℝᵈ。训练仅依赖用户最近 M 条动作序列输入。

**正反馈/Positive Engagement**

关注能体现强兴趣的三类 Homefeed 行为：

- 保存（Repin）
- 放大查看 > 10 秒（Close-up）
- 外链点击 > 10 秒（Long Click）

**训练任务/Model Training**

非传统 “Next Action Prediction”，而是让生成的用户向量 u 在之后 **14 天** 内能优先靠近其可能正反馈的 Pin 向量 p。

- 系统每天离线为用户生成一次向量 **u**（基于其最近 M 条行为）。
- 若距离 d(u, pᵢ) < d(u, pⱼ)，则假设用户更可能在未来两周内与 pᵢ 产生正交互。
- 14 天窗口平衡了可计算性与长期兴趣刻画。（Pinterest 观察到两周足以覆盖大部分用户的灵感周期，能体现**较长期**的偏好）

**3.1 Feature Encoding**

用户行为特征刻画

- PinSage embedding （行为图网络学习出的256维向量）
- Metadata
    - **行为类型/action type**（保存、点击等）
    - **展示面/surface**（Homefeed、Search 等）
    - **时间戳/timestamp**
    - **行为时长/action duration**

行为类型和surface以embedding形式编码，而行为时长作为一个标量特征以log(duration)加工。

- **时间编码**

除原始时间戳外，生成新的feature：

- 与最新一次行为的时间间隔
- 相邻两次行为的间隔

对每个时间特征，采用类 Time2Vec 的方法。

将 PinSage 嵌入、类别嵌入、时长特征及时间向量**拼接。**

**3.2 Model Architecture**

PinnerFormer 选择双塔架构

![image.png](attachment:c20cbe08-9db6-4255-a1bb-5ffdecd69f49:image.png)

- 用户塔（User Tower）
    - Transformer + Pre-Norm；行为特征拼接后 → 位置编码 → MHSA/FFN → MLP → L2 归一化
    - 注意transformer做了causal mask：保证了第 *i* 步只能利用过去的信息来生成表示。
- 内容塔（Pin Tower）
    - 仅取 PinSage 256 维向量 → MLP → L2 归一化

**3.3 Metric Learning**

训练用户嵌入 u 与 Pin 嵌入 p，使 “用户-正样本” 比 “用户-负样本” 更相似。

- 负样本选择/Negative Selection
    - in-batch negatives： 同batch中其他用户的正样本当作当前用户的负样本。
        - 优点是*无需额外采样*，实现简单
        - 缺点是热门 Pin 更易被当作负样本 → 可能误降权；分布与线上检索不一致
    - random negatives：从全量候选 Pin 中均匀采样
        - 优点是分布接近线上真实库
        - 缺点是过于容易，易导致模型塌缩
    - 合并in-batch negatives + random negatives
- 损失函数/Loss Function
    - sampled softmax loss
        - 仅对一个正样本pi 和N 个负样本计算 softmax。
        - 若负样本并非均匀采样，需要 **log Q 校正。**
            - 𝑄𝑖(𝑣) = 𝑃 (Pin 𝑣 in batch | User 𝑈𝑖 in batch)

![image.png](attachment:e8e7f901-a724-4c89-bfac-e13a0a938687:image.png)

**3.4 Training Objective**

![image.png](attachment:10f5c3f2-c1b6-43d2-93e6-b70ba21066d7:image.png)

Four training objectives considered:

- **下一步预测 (Next Action)：** 大多数序列建模模型的任务目标，只用用户序列 {A_T … A_{T-M+1}} 去预测下一条正向行为 A_{T+1}； 但不能充分捕捉用户长期兴趣；
- **SASRec：** 对序列中每个时间步都预测其下一条正向行为，缺点同上；
- **All Action Prediction：** 用最终用户嵌入 e₁ 预测未来 **K 天** 内 *全部* 正向行为
- **Dense All Action Prediction**
    - 训练对齐： 随机选若干历史位置 {sᵢ}，对每个位置输出 e_{sᵢ} 预测其后 **K 天** 内随机一条正向行为；
    - Transformer 施加 **因果掩码；**

与 **All Action Prediction**（只用最终 e₁ 预测窗口中的全部正样本）相比，**Dense All Action** 让 **序列里的多个位置都参与预测**，因此称 “Dense”。

**3.5 Dataset Design**

实验可调超参数

- 用户最大序列长度 M
- 从用户时间线抽取的序列窗口比例
- 每位用户最多采样序列数量
- 每条序列最多采样正样本数量

**3.6 Model Serving**

![image.png](attachment:6d3d3a86-9ece-4b0b-a90e-1597996142be:image.png)

**日更增量**

- 每天离线推断 **PinnerFormer 用户embedding**，**仅**为过去 24 小时内有新交互的用户重算嵌入；
- 将新向量与前一日embedding合并后上传至 **Key-Value 特征库**，供线上实时读取。

**Pin embeddings**

- 计算成本低：只对现有 PinSage 向量过一层小 MLP；
- 每日全量重算，并构建 **HNSW 近邻索引**，线上用当天用户向量快速检索候选 Pin。
