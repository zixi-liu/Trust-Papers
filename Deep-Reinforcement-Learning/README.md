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

**Two types of Value-based Methods**
- The goal of an RL agent is to have an optimal policy π*.
  - use an Epsilon-Greedy Policy that handles the exploration/exploitation trade-off
- The state-value function


