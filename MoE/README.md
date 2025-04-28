## 综述

- [Mixture of Experts（MoE）学习笔记](https://zhuanlan.zhihu.com/p/675216281)
  - MoE通过weight function计算一个权重来将任务给到具体的模型来解决问题, 将一个大的问题空间拆分成小的子空间交由不同Expert解决。
  - [**Deep Mixture of Experts**](https://arxiv.org/pdf/1312.4314)
  - [**Sparsely-gated MoE layer**](https://arxiv.org/pdf/1701.06538)
    - <img width="601" alt="image" src="https://github.com/user-attachments/assets/9bc8eaa3-d514-4c88-b0c5-ec00bc0a111b" />
  - [**Noisy Top-K Gating**]
  - [**Balancing Expert Utilization**]
  - MoE与Transformer
    - ST-MoE（Stable and Transferable）

## [月球大叔EP07]

[[FIXME][EP07] 聊聊MoE + 闲谈学术品位](https://www.youtube.com/watch?v=mHUBwzlsWjg)

**一个MoE的常见误区：**

- MoE中的Experts并不是指多个独立的模型，而是指在同一个模型内部的多个子层（子模块），体现的是功能网络（FN）分层中稀疏激活的思想。每个token输入后，首先通过router分配，然后被送往相应的expert进行处理。

***Switch Transformer Illustration***

![image](https://github.com/user-attachments/assets/5b9c4409-daff-4743-a200-58ee11ffad64)

Router的作用

- 本质上是一个linear layer， 作用是把输入的token表示乘上一个权重矩阵W（点积），输出一组**scores，**选择得分最高的1个或几个expert，让这个token被送过去处理。

## History of MoE

**2017: LSTM+MoE**

- Sparsely Gated: 这里的gating，属于离散操作，不可微分(non-differentiable)。
- 作者引入**Noisy Top-K Gating**实现近似的可微分。

![image](https://github.com/user-attachments/assets/c9fedc24-d552-4d84-8688-c2d75e80df26)

**2021: Gshard (MoE MLSys)**

- MoE是在不增加计算量的情况下，增加参数量。
- 思想是将MoE中的expert放到不同的GPU上面。

**2021: Switch Transformer (MoE Transformer)**

**GLaM, ST-MoE…**

- [awesome-mixture-of-experts](https://github.com/XueFuzhao/awesome-mixture-of-experts)

**2023: OpenMoE → A family of open-sourced MoE models, Mixtral**

**2024: DeepSeek-MoE, OLMoE**

**2025: LLaMA-4…**

关于Training MoE：

- MoE不是很好训练，Scaling没有免费的午餐。但站在巨人的肩膀上，有很多提升Scalability的trick。

## Mixture-of-Experts

![image](https://github.com/user-attachments/assets/a606cdf3-7030-4acc-bb77-6a8f40aafd5e)

- 通过条件激活子网络（Experts）实现计算稀疏化。每个输入token *x*被Router动态地分配给一组Experts处理。
- G是linear layer，presenter提到尝试了一些将G scale up的工作，但没有positive signal。

Sparse Routing

![image](https://github.com/user-attachments/assets/71bd52c9-875e-4e4c-ae89-d0155ae3eef6)

- s(x): router打的原始分数
- ϵ(x)∼N(0,σ2)Gaussian噪声鼓励exploration of expert routing。

**Balanced Loading**

- 某个expert收到太多tokens（容易超负荷，导致延迟）。
- 某个expert收到的tokens太少（资源浪费）。

为了解决第一个问题（防止单个expert超载），定义了一个**buffer capacity B**
- B=CKNL

其中：

- C：容量系数（capacity ratio）。
- K：每个token要选择的expert数量。
- N：每个设备上的batch size。
- L：序列长度。

训练时引入了一个**辅助损失（Auxiliary Loss）**，叫做**均衡损失（**lbalance**）。**让实际的token分配比例和router的softmax概率都趋近于**均匀分布**（即每个expert负载接近均匀）。

## OpenMoE

### Analyzing OpenMoE - **通过可视化理解MoE**

- 1. **Does MoE specialize in domain level？**
  - expert在domain level没有显著的specialization。
- 2. **Does** **MoE specialize in language level？**
  - 尝试了编程语言和自然语言。自然语言上有一些specialization。
- 3. **Does** **MoE specialize in task level？**
  - 有一些specialization，但不是特别明显。
- 4. **Does MoE specialize in Position ID？**
  - Answer：No
- 5. **Does MoE specialize in Token ID？**
  - 尽管同一个token在不同句子里出现时语境各不相同，但模型依然倾向于把它们路由到很少数固定的experts。
  - 并没有学到high-level semantics，而是基于Token ID简单的做routing。

### **Early Routing Learning**

- Routing Behavior一般会很早固定，因为模型会尝试先优化load balance loss，再去学next token prediction。
- 但是模型早期并没有high-level semantics，所以模型没有办法根据high-level semantics学习routing。

### Drops Towards the End

- 为了让每个expert处理的tokens数量比较均衡，通常会为每个expert设置一个最大容量 C。每个expert最多只能处理C个tokens，超了就丢弃(drop)。这种后面的tokens更容易被丢掉的问题，叫Drop-towards-the-End。
- 在多语言数据集和指令跟随数据集（instruction-following datasets，比如chat data）上，后面出现的tokens掉得特别多。（OpenAI建议把prompt写在前面）

![image](https://github.com/user-attachments/assets/10e0ac01-9057-4245-bf3b-3574c2ec98dd)

## Important Recent Work

- Megablock
    - 针对不设capacity的一种优化。尽量少的使用expert parallel → 将dense矩阵转换为使用稀疏矩阵运算。
    - ![image](https://github.com/user-attachments/assets/d67a2801-b3c4-4e0b-8b4b-0f37eccb65e3)
- DeepSeek MoE
    - 细粒度routing - expert很多但每个expert比较小，且激活expert个数比较多。routing更加smooth。
    - 增加共享expert。提高inference MFU。
    - ![image](https://github.com/user-attachments/assets/083c101b-3693-4e9b-9f15-d7a2baf12731)

## More Thoughts

- MoE is tricky to train (load balance, training stability)
- MoE is more data hungry.
    - MoE is easier to overfit when using limited and repeated data.
- MoE is more sensitive to data diversity.
    - Deploying MoE foundation model on different countries.
- Where shall we place the MoE layers?
   - ![image](https://github.com/user-attachments/assets/6841e754-2fb2-4980-b854-f305d3173d59)
- Which routing algorithm is better?
  - ![image](https://github.com/user-attachments/assets/034404d7-a2d2-46ba-9d0a-76d3cbac86fd)

## **Q&As**

问答中一些有趣的topics

- prefetch expert
