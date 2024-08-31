# Applying Supervised Learning Concepts to Reinforcement Learning

Coming from a deep learning background, I was familiar with the effectiveness of the gradient descent algorithm in optimizing neural networks for supervised learning tasks. This made me wonder: why is reinforcement learning (RL) treated so differently, and why can't we just apply gradient descent directly to solve RL tasks?

Initially, I explored the idea of using the reward as a cost function and applying gradient ascent to optimize a neural network in RL settings. However, I quickly realized a fundamental flaw in this approach. In RL, the reward does not directly provide information about the gradient of the action space. The reward function often depends on a sequence of actions over time rather than a direct mapping from the network's output to an immediate reward. This makes it challenging, if not impossible, to compute the gradient with respect to the actions using standard backpropagation techniques. If we set the cost as the reward and directly apply gradient ascent, there’s no mechanism to explicitly select actions or learn the probability distribution over actions.

After further research, I discovered that the correct way to approach this problem, which had already been established in 1992 by R.J. Williams, is through a method called the REINFORCE algorithm. This algorithm provides a more structured and theoretically sound way to apply gradient-based optimization to RL tasks by estimating gradients based on the expected return rather than direct supervision.

The **REINFORCE** algorithm remains a foundational technique in reinforcement learning. It demonstrates that while RL tasks can indeed benefit from gradient-based optimization, this requires modifications to handle the unique aspects of RL, such as the temporal nature of rewards and the need for exploration.

## Algorithm Equations

The equations I derived myself to prove that gradient ascent was possible are in the ```images``` folder, albeit they aren't perfect.

## Experiments

To test my understanding, I compared my initial naive algorithm with the REINFORCE algorithm on the CartPole environment, a standard benchmark in reinforcement learning. The results clearly showed the advantages of using policy gradients as implemented in REINFORCE, highlighting its ability to solve RL tasks that my initial approach could not.

## Acknowledgments
Special thanks to R.J. Williams for introducing the REINFORCE algorithm in 1992, which laid the groundwork for applying gradient-based optimization to reinforcement learning.
