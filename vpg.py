import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

class VanillaPolicyGradient:
    def __init__(self, env, policy_network, learning_rate=1e-3):
        self.env = env
        self.policy_network = policy_network
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def compute_returns(self, rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def train(self, num_episodes=1000, gamma=0.99):
        all_rewards = []
        avg_rewards = []

        for episode in range(num_episodes):
            state, info = self.env.reset()
            log_probs = []
            rewards = []
            done = False
            
            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state
                
                done = terminated or truncated
            
            all_rewards.append(sum(rewards))
            avg_rewards.append(np.mean(all_rewards[-100:]))  # Calculate average reward over last 100 episodes

            returns = self.compute_returns(rewards, gamma)
            loss = 0
            
            for log_prob, R in zip(log_probs, returns):
                loss -= log_prob * R
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if episode % 50 == 0:
                print(f"Episode {episode}, Total Reward: {sum(rewards)}, Average Reward (last 100): {avg_rewards[-1]}")
            
            # Check if the environment is solved
            if avg_rewards[-1] >= 195.0 and episode >= 100:
                print(f"Solved! Episode {episode}, Average Reward: {avg_rewards[-1]}")
                break
        
        return all_rewards, avg_rewards