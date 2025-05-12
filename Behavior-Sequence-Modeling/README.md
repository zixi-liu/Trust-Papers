
**Pinterest用户兴趣建模**
- [[KDD'22] PinnerFormer: Sequence Modeling for User Representation at Pinterest](https://arxiv.org/pdf/2205.04507)
  - two key challenges:
    - (a) cost of computation
    - (b) infrastructure complexity
  - learns embedding to capture a user’s longer-term interests, rather than only predicting the next action. 
  - develop a single high quality user embedding that can be used for many downstream tasks.
  - design choices:
    - Single vs. multiple embeddings for a single user
  - [KDD'22|天级更新超越实时](https://zhuanlan.zhihu.com/p/558608369)
    - 建模一个高质量的用户长期兴趣表达，给业务所有场景使用。
    - 提出dense all action训练目标，相比predict next的建模方式，可以学出高质量的用户长期兴趣表达。
    - 行为序列特征：
      - PinSage embedding
      - meta features，包括用户行为类型，surface，时间戳和行为时长
      - 自用户最近一次行为以来的时间间隔，以及行为之间的时间间隔。Time2Vec
    
      
