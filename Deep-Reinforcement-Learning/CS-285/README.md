
### Intro

**What does reinforcement learning do differently?**

Deep reinforcement learning
- Classical reinforcement learning

<img width="390" alt="image" src="https://github.com/user-attachments/assets/478d7bc2-6819-420d-97f8-cb75e468fb1c" />

- Evolutionary algorithms, controls, optimization


Reinforcement learning
- assumes data is not i.i.d.: previous outputs influence future inputs!
- ground truth answer is not known, only known if we succeeded or failed
  - more generally, we know the reward

### Imitation

Terminologies
- the actions you choose will affect the observations you see in the future

Behavioral Clone


### Training Policies

**Reward Functions**

- r(s, a): tells us which states and actions are better

**Algorithms**
- generate samples
  - run the policy
- fit a model/estimate the return
- improve the policy


