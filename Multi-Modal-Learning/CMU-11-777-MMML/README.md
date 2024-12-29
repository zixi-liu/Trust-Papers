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

