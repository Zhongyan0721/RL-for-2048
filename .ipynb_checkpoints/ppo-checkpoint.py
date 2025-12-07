import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        # Input is (4, 4, 1) -> but we might want to one-hot encode it or just flatten.
        # Let's assume we preprocess the input to be log2(x) and normalize, or use layers.
        # Let's use a simpler approach: Flatten -> Dense layers.
        
        # If we treat input as simple float values (normalized log2)
        self.network = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten
        features = self.network(x)
        return self.actor(features), self.critic(features)

class PPO:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Input dimension: 4x4 = 16
        self.policy = ActorCritic(16, 4).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.policy_old = ActorCritic(16, 4).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device) # Shape (4, 4, 1)
            state = state.view(1, -1) # Flatten to (1, 16)
            # Preprocessing: log2
            state = torch.log2(state + 1) # +1 to handle 0
            
            logits, _ = self.policy_old(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action).item()

    def update(self, memory):
        # Convert list to tensor
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # Normalize rewards
        if rewards.std() > 1e-5:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        else:
            rewards = rewards - rewards.mean()

        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.tensor(memory.actions).to(self.device).detach()
        old_logprobs = torch.tensor(memory.logprobs).to(self.device).detach()
        
        # Preprocess states
        old_states = old_states.view(old_states.size(0), -1)
        old_states = torch.log2(old_states + 1)

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logits, state_values = self.policy(old_states)
            dist = Categorical(logits=logits)
            
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = state_values.squeeze()
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Loss
            loss = -torch.min(surr1, surr2) + 0.5*self.mse_loss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

