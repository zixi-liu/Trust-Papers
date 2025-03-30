## Deep Reinforcement Learning

### Unit 1. Introduction to Deep Reinforcement Learning

**Concepts**
- An agent (an AI) will learn from the environment by interacting with it (through trial and error) and receiving rewards (negative or positive) as feedback for performing actions.
- The agent’s goal is to maximize its cumulative reward, called the expected return.
- Markov Property: our agent needs only the current state to decide what action to take.
- Observation/States Space:
  - State s: is a complete description of the state of the world.
  - Observation o: is a partial description of the state.
- Action space is the set of all possible actions in an environment.
- Rewards and the discounting
- Episodic task and continuing tasks
- The Exploration/Exploitation trade-off
  - Exploration is exploring the environment by trying random actions in order to find more information about the environment.
  - Exploitation is exploiting known information to maximize the reward.

**Policy-Based Methods**
- By teaching the agent to learn which action to take, given the current state.
- Deterministic: a policy at a given state will always return the same action.
- Stochastic: outputs a probability distribution over actions.

**Value-based Methods**
- Learn a value function that maps a state to the expected value of being at that state.
- The value of a state is the expected discounted return the agent can get if it starts in that state, and then acts according to our policy.

### Unit 2 Intro to Q-Learning

**Two types of Value-based Methods**
- The goal of an RL agent is to have an optimal policy π*.
  - use an Epsilon-Greedy Policy that handles the exploration/exploitation trade-off
- The state-value function
  - outputs the expected return if the agent starts at that state and then follows the policy forever afterward
- The action-value function
  - outputs the expected return if the agent starts in that state, takes that action, and then follows the policy forever after.

**The Bellman equation**
- a recursive equation that works like this: instead of starting for each state from the beginning and calculating the return, we can consider the value of any state as:
  - The immediate reward + the discounted value of the state that follows
 
**Monte Carlo vs Temporal Difference Learning**

- Monte Carlo: learning at the end of the episode
- Temporal Difference, on the other hand, waits for only one interaction (one step) to form a TD target and update.

**Q-Learning**
- Q-Learning is an off-policy value-based method that uses a TD approach to train its action-value function.
- Q-function is encoded by a Q-table, a table where each cell corresponds to a state-action pair value.
- [A Q-Learning Example](https://huggingface.co/learn/deep-rl-course/en/unit2/q-learning-example)

**Off-policy vs on-policy algorithms**
- Off-policy algorithms: A different policy is used at training time and inference time
- On-policy algorithms: The same policy is used during training and inference

### Unit 3 Deep Q-Learning 

**The Deep Q-Network (DQN)**

Preprocessing the input and temporal limitation
- We stack frames together because it helps us handle the problem of temporal limitation. 

*Deep Q-Learning uses a neural network to approximate, given a state, the different Q-values for each possible action at that state.*
 
**The Deep Q-Learning Algorithm**

Deep Q-Learning uses a deep neural network to approximate the different Q-values for each possible action at a state (value-function estimation).
- In Deep Q-Learning, we create a loss function that compares our Q-value prediction and the Q-target and uses gradient descent to update the weights of our Deep Q-Network to approximate our Q-values better

The Deep Q-Learning training algorithm has two phases:
- Sampling: we perform actions and store the observed experience tuples in a replay memory.
- Training: Select a small batch of tuples randomly and learn from this batch using a gradient descent update step.

Deep Q-Learning training might suffer from instability, mainly because of combining a non-linear Q-value function (Neural Network) and bootstrapping (when we update targets with existing estimates and not an actual complete return).

To help us stabilize the training, we implement three different solutions:
- *Experience Replay* to make more efficient use of experiences.
  - Use a replay buffer that saves experience samples that we can reuse during the training. This allows the agent to learn from the same experiences multiple times.
  - Avoid forgetting previous experiences (aka catastrophic interference, or catastrophic forgetting) and reduce the correlation between experiences. By randomly sampling the experiences, we remove correlation in the observation sequences and avoid action values from oscillating or diverging catastrophically.
- *Fixed Q-Target* to stabilize the training.
  - When we calculate the loss, we calculate the difference between the TD target (Q-Target) and the current Q-value (estimation of Q). To estimate TD target, we use the Bellman equation - we saw that the TD target is just the reward of taking that action at that state plus the discounted highest Q value for the next state.
  - To avoid significant oscillation in training, we use a separate network with fixed parameters for estimating the TD Target. We copy the parameters from our Deep Q-Network every C steps to update the target network.
- *Double Deep Q-Learning*, to handle the problem of the overestimation of Q-values.
  - If non-optimal actions are regularly given a higher Q value than the optimal best action, the learning will be complicated.
  - When we compute the Q target, we use two networks to decouple the action selection from the target Q-value generation.
    - Use our DQN network to select the best action to take for the next state (the action with the highest Q-value).
    - Use our Target network to calculate the target Q-value of taking that action at the next state.

### Unit 4 Policy Gradient with PyTorch

**What are the policy-based methods?**

Value-based, Policy-based, and Actor-critic methods

In value-based methods, we learn a value function.
- Our objective is to minimize the loss between the predicted and target value to approximate the true action-value function.
- We have a policy, but it’s implicit since it is generated directly from the value function. For instance, in Q-Learning, we used an (epsilon-)greedy policy.

In policy-based methods, we directly learn to approximate without having to learn a value function.
- The idea is to parameterize the policy. For instance, using a neural network, this policy will output a probability distribution over actions (stochastic policy).
- Our objective then is to maximize the performance of the parameterized policy using gradient ascent.

Policy-based methods vs policy-gradient methods
- In policy-based methods, we search directly for the optimal policy. We can optimize the parameter θ indirectly by maximizing the local approximation of the objective function with techniques like hill climbing, simulated annealing, or evolution strategies.
- In policy-gradient methods, because it is a subclass of the policy-based methods, we search directly for the optimal policy. But we optimize the parameter θ directly by performing the gradient ascent on the performance of the objective function J(θ).

Policy-gradient methods can learn a stochastic policy
- We don’t need to implement an exploration/exploitation trade-off by hand. Since we output a probability distribution over actions, the agent explores the state space without always taking the same trajectory.
- We also get rid of the problem of perceptual aliasing. Perceptual aliasing is when two states seem (or are) the same but need different actions.

Value-based Reinforcement learning algorithm vs an optimal stochastic policy
- Under a value-based Reinforcement learning algorithm, we learn a quasi-deterministic policy (“greedy epsilon strategy”). Consequently, our agent can spend a lot of time before finding the dust.
- On the other hand, an optimal stochastic policy will randomly move left or right in red (colored) states. Consequently, it will not be stuck and will reach the goal state with a high probability.

Policy-gradient methods are more effective in high-dimensional action spaces and continuous actions spaces

**Diving deeper into policy-gradient methods**

Our goal with policy-gradient is to control the probability distribution of actions by tuning the policy such that good actions (that maximize the return) are sampled more frequently in the future. 

For each state-action pair, we want to increase the P(a∣s): the probability of taking that action at that state. Or decrease if we lost.

### Unit 5 Intro to Unity ML-Agents






### Resources

- [深度强化学习发展史](https://zhuanlan.zhihu.com/p/56399184)
