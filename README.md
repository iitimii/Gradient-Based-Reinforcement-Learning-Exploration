# Applying Supervised Learning to Reinforcement Learning Cases

Coming from a deep learning background, I was accustomed to the power of the gradient descent algorithm. I've always wondered, what's all the fuss about reinforcement learning (RL), and why can't we just use gradient descent to solve RL tasks? I explored the possibility of using the reward as the cost function and applying the gradient ascent algorithm. After some brainstorming and experimentation, I put this to practice. My algorithm is flawed because the reward ia a scalar, and the reward fucntion is not directly dependent on the actions of the neural network, thus impossible to find the derivative.

A correct and more elaborate implementation of this algorithm i was trying to develop (as I later found out it already existed, R.J Wlliams 1992) is known as the REINFORCE algorithm.

This algorithm, called **REINFORCE**, remains a foundational technique in reinforcement learning. It demonstrates that RL tasks can indeed benefit from gradient-based optimization, albeit with some modifications to handle the unique challenges of RL.

## Algorithm Equations

The equations I derived myself to prove that gradient ascent was possible are in the ```images``` folder, albeit they aren't perfect.

## Experiments

I applied both my simple algorithm and the standard REINFORCE algorithm to solve the cartpole reinforcement learning environment.

These experiments showcase the effectiveness of using policy gradients, a form of gradient ascent, for solving a range of RL tasks.

## Acknowledgments
- R.J. Williams for introducing the REINFORCE algorithm in 1992.
