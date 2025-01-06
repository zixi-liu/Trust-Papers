[Foundations & Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions](https://arxiv.org/pdf/2209.03430)

## Introduction

#### Core multimodal technical challenges

**Challenge 1. Representation**
- Learning representations that reflect cross-modal interactions between individual elements, across different modalities.
  - Fusion (# modalities > # representations)
    - Fusion with abstract modalities: suitable unimodal encoders are first applied to capture a holistic representation of each element (or modality entirely)
      - Additive and multiplicative interactions
      - Multiplicative Interactions (MI) generalize additive and multiplicative operators to include learnable parameters that capture second-order interactions.
    - Fusion with raw modalities (early stage)
  - Coordination (# modalities = # representations)
    - **strong coordination** that enforces strong equivalence between modality elements (to bring semantically corresponding modalities close together in a coordinated space)
    - **partial coordination** captures more general modality connections such as correlation, order, hierarchies, or relationships.
  - Fission (# modalities < # representations)
    - **Modality-level fission** aims to factorize into modality-specific information primarily in each modality and multimodal information redundant in both modalities.
    - **Fine-grained fission** further break multimodal data down into the individual subspaces covered by the modalities.

**Challenge 2. Alignment**
- Identifying and modeling cross-modal connections between all elements of multiple modalities, building from the data structure.
  - Discrete Alignment
    - **local alignment** to discover connections between a given matching pair of modality elements.
    - **global alignment** where alignment must be performed globally to learn both the connections and matchings.
  - Continuous Alignment
    - **Continuous warping** aims to align two sets of modality elements by representing them as continuous representation spaces and forming a bridge between these representation spaces.
      - Adversarial training is a popular approach to warp one representation space into another.
      - Dynamic time warping (DTW) is a related approach to segment and align multi-view time series data.
    - **Modality segmentation** involves dividing high-dimensional data into elements with semanticallymeaningful boundaries.
      - A common problem involves temporal segmentation, where the goal is to discover the temporal boundaries across sequential data.
  - Contextualized Representation
    - **Joint undirected alignment** aims to capture undirected connections across pairs of modalities, where the connections are symmetric in either direction.
    - **Cross-modal directed alignment** relates elements from a source modality in a directed manner to a target modality, which can model asymmetric connections.
    - **Graphical alignment** that generalizes the sequential pattern into arbitrary graph structures.

**Challenge 3. Reasoning**

**Challenge 4. Generation**

**Challenge 5. Transference**

Transfer knowledge between modalities, usually to help the target modality which may be noisy or with limited resources.
- Cross-modal Transfer
  - Tuning
  - Multitask learning aims to use multiple large-scale tasks to improve performance as compared to learning on individual tasks.
  - Transfer learning
- Multimodal Co-learning
- Model Induction

**Challenge 6. Quantification**

## Unimodal Representations

## Multimodal Representation Fusion

**Cross-modal Interactions**
- Interactions happen during inference!
  - Representation fusion
  - Prediction task
  - Modality translation
- Interactions: How multimodal information changes when modalities are combined for a response.
- Taxonomy of Interaction Responses
  - Redundancy
    - Equivalence
    - Enhancement
  - Nonredundancy
    - Independence
    - Dominance
    - Modulation
    - Emergence

#### Sub-Challenge 1a: Representation Fusion

Learn a joint representation that models cross-modal interactions between individual elements of different modalities
- Basic fusion (homogeneous) - pre-trained encoders
- Raw-modality fusion (heterogeneous)

**Early and Late Fusion**

**Basic Concepts for Representation Fusion (Basic Fusion)**
- univariate case
  - ![image](https://github.com/user-attachments/assets/ab5dc0d0-8490-41fb-a2ed-c857c2378e95)
- Tensor Fusion
- Low-rank Fusion
- Gated Fusion
  - ![image](https://github.com/user-attachments/assets/31e40a8a-f1c2-4b77-9c59-45785b8f7d2d)
  - ga and gb can be seen as attention functions
- Modality-shifting Fusion
- Mixture of Fusions (gating can be soft or hard attention)
- Nonlinear Fusion

**Fusion with Heterogeneous Modalities**
- Multimodal Masked Autoencoder
- Dynamic Early Fusion
  - Define basic representation building blocks
  - Define basic fusion building blocks
  - Automatically search for composition using neural architecture search
- Heterogeneity-aware Fusion

**Improving Optimization**
- sometimes multimodal doesn’t help
  - Multimodal networks are more prone to overfitting due to increased complexity.
  - Different modalities overfit and generalize at different rates.

**Heterogeneity in Noise: Studying Robustness**
- Noise within Modality
- Missing Modalities
- Strong tradeoffs between performance and robustness

Several approaches towards more robust models
- Robust data + training
- Infer missing modalities

![image](https://github.com/user-attachments/assets/0e63d684-c191-46c5-bf60-9dac08d1a9a6)

## Multimodal Alignment

**Heterogenous, connected and interacting data**
- Connections
  - knowledge of one modality provides information about the other modality
  - connection types
    - co-occurrence
    - correlation
    - causality
- Coordinated Representations - Example: CLIP focuses on shared connections
- Modality Interactions
  - Interaction with a response (inference)
  - Interactions taxonomy:
    - Level 1: Responses and Input Modalities
      - Co-occurrence
      - Redundancy
      - Dominance
      - Emergence
    - Level 2: Interactions - Internal Mechanics
      - Additive
      - Multiplicative
      - Polynomial
      - Nonlinear
    - Level 3: Contextualized Interactions
      - Temporal
      - Hierarchy
      - Multimodal
  - Responses and Input Modalities
    - Information theory as a framework
      - 

## Other Resources

[该怎样去学习多模态（表征、翻译、对齐、融合等）？](https://www.zhihu.com/question/638854224/answer/3408335048)
