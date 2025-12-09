import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import math

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Game2048Encoder(nn.Module):
    """
    Custom CNN encoder with directional kernels for 2048.
    Uses depthwise separable convolutions with:
    - 1x4 horizontal kernels (row patterns)
    - 4x1 vertical kernels (column patterns)  
    - 4x4 global kernels (full board)
    
    Input: One-hot encoded board (16, 4, 4)
    Output: Feature vector (1024,)
    """
    def __init__(self, num_tile_types=16, channel_multiplier=16):
        super().__init__()
        in_channels = num_tile_types
        mid_channels = in_channels * channel_multiplier  # 16 * 16 = 256
        
        # Horizontal branch: 1x4 depthwise conv captures row patterns
        self.horizontal_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 4), 
                                       groups=in_channels, bias=False)
        self.horizontal_pw = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        
        # Vertical branch: 4x1 depthwise conv captures column patterns
        self.vertical_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(4, 1),
                                     groups=in_channels, bias=False)
        self.vertical_pw = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        
        # Global branch: 4x4 depthwise conv captures full board
        self.global_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(4, 4),
                                   groups=in_channels, bias=False)
        self.global_pw = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        
        self.activation = nn.LeakyReLU(0.1)
        
        # After convolutions:
        # horizontal: (batch, 256, 4, 1) -> 4 spatial positions
        # vertical: (batch, 256, 1, 4) -> 4 spatial positions
        # global: (batch, 256, 1, 1) -> 1 spatial position
        # Total: 4 + 4 + 1 = 9 positions (or with padding could be 12)
        
        # Final conv to produce 1024-dim feature
        # We'll flatten and use a linear layer for flexibility
        self.final_fc = nn.Linear(mid_channels * 9, 1024)
        
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for all weights, zero for biases."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x: (batch, 16, 4, 4) one-hot encoded
        
        # Horizontal branch
        h = self.horizontal_dw(x)  # (batch, 16, 4, 1)
        h = self.activation(h)
        h = self.horizontal_pw(h)  # (batch, 256, 4, 1)
        h = self.activation(h)
        h = h.view(h.size(0), -1)  # (batch, 256*4)
        
        # Vertical branch
        v = self.vertical_dw(x)    # (batch, 16, 1, 4)
        v = self.activation(v)
        v = self.vertical_pw(v)    # (batch, 256, 1, 4)
        v = self.activation(v)
        v = v.view(v.size(0), -1)  # (batch, 256*4)
        
        # Global branch
        g = self.global_dw(x)      # (batch, 16, 1, 1)
        g = self.activation(g)
        g = self.global_pw(g)      # (batch, 256, 1, 1)
        g = self.activation(g)
        g = g.view(g.size(0), -1)  # (batch, 256)
        
        # Concatenate all branches
        features = torch.cat([h, v, g], dim=1)  # (batch, 256*9)
        features = self.final_fc(features)       # (batch, 1024)
        features = self.activation(features)
        
        return features


class ActorCritic(nn.Module):
    """
    Actor-Critic network with custom CNN encoder for 2048.
    
    Architecture from report:
    - Shared encoder: Custom CNN with directional kernels -> 1024 features
    - Actor head: 1024 -> 256 -> 64 -> 4 (action logits)
    - Critic head: 1024 -> 256 -> 64 -> 1 (state value)
    """
    def __init__(self, num_tile_types=16, num_actions=4):
        super().__init__()
        
        # Shared feature extractor
        self.encoder = Game2048Encoder(num_tile_types=num_tile_types)
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._init_heads()
    
    def _init_heads(self):
        """Orthogonal initialization for actor and critic heads."""
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: One-hot encoded state (batch, 16, 4, 4)
        Returns:
            action_logits: (batch, 4)
            state_value: (batch, 1)
        """
        features = self.encoder(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value
    
    def get_action(self, x, action=None, valid_mask=None):
        """
        Get action, log probability, entropy, and value.
        
        Args:
            x: One-hot encoded state
            action: If provided, compute log_prob for this action
            valid_mask: Boolean mask for valid actions (batch, 4)
        """
        action_logits, value = self.forward(x)
        
        # Apply valid action mask if provided
        if valid_mask is not None:
            # Set logits of invalid actions to -inf
            action_logits = action_logits.masked_fill(~valid_mask, float('-inf'))
        
        dist = Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


def one_hot_encode(grid, num_tile_types=16):
    """
    Convert raw grid to one-hot encoding.
    
    Args:
        grid: (batch, 4, 4) or (4, 4) with raw tile values (0, 2, 4, 8, ...)
    
    Returns:
        one_hot: (batch, 16, 4, 4) one-hot encoded
    """
    if grid.dim() == 2:
        grid = grid.unsqueeze(0)
    
    batch_size = grid.size(0)
    device = grid.device
    
    # Convert tile values to ranks: 0->0, 2->1, 4->2, 8->3, ...
    # log2(x) for x > 0, 0 for empty cells
    ranks = torch.zeros_like(grid, dtype=torch.long)
    nonzero_mask = grid > 0
    ranks[nonzero_mask] = torch.log2(grid[nonzero_mask].float()).long()
    
    # Clamp to valid range [0, num_tile_types-1]
    ranks = torch.clamp(ranks, 0, num_tile_types - 1)
    
    # Create one-hot encoding
    one_hot = torch.zeros(batch_size, num_tile_types, 4, 4, device=device)
    
    # Scatter ones at appropriate positions
    for b in range(batch_size):
        for i in range(4):
            for j in range(4):
                one_hot[b, ranks[b, i, j], i, j] = 1.0
    
    return one_hot


def one_hot_encode_fast(grid, num_tile_types=16):
    """
    Fast vectorized one-hot encoding.
    
    Args:
        grid: (batch, 4, 4) with raw tile values
    
    Returns:
        one_hot: (batch, 16, 4, 4)
    """
    if grid.dim() == 2:
        grid = grid.unsqueeze(0)
    
    batch_size = grid.size(0)
    device = grid.device
    
    # Convert to ranks
    ranks = torch.zeros_like(grid, dtype=torch.long)
    nonzero_mask = grid > 0
    if nonzero_mask.any():
        ranks[nonzero_mask] = torch.log2(grid[nonzero_mask].float()).long()
    ranks = torch.clamp(ranks, 0, num_tile_types - 1)
    
    # Flatten spatial dimensions
    ranks_flat = ranks.view(batch_size, -1)  # (batch, 16)
    
    # Create one-hot using scatter
    one_hot_flat = torch.zeros(batch_size, 16, num_tile_types, device=device)
    one_hot_flat.scatter_(2, ranks_flat.unsqueeze(-1), 1.0)
    
    # Reshape to (batch, num_tile_types, 4, 4)
    one_hot = one_hot_flat.permute(0, 2, 1).view(batch_size, num_tile_types, 4, 4)
    
    return one_hot


class PPO:
    """
    Proximal Policy Optimization with GAE.
    
    Hyperparameters from report:
    - gamma = 0.997
    - gae_lambda = 0.9
    - eps_clip = 0.1
    - actor_lr = 2.5e-5
    - critic_lr = 6.25e-5
    - entropy_coef = 2.5e-4
    - critic_coef = 1e-8
    """
    def __init__(self, env, 
                 actor_lr=2.5e-5, 
                 critic_lr=6.25e-5,
                 gamma=0.997, 
                 gae_lambda=0.9,
                 eps_clip=0.1, 
                 k_epochs=4,
                 entropy_coef=2.5e-4,
                 critic_coef=1e-8):
        
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor-Critic network
        self.policy = ActorCritic().to(self.device)
        
        # Separate optimizer parameter groups for actor and critic
        actor_params = list(self.policy.encoder.parameters()) + list(self.policy.actor.parameters())
        critic_params = list(self.policy.critic.parameters())
        
        self.optimizer = optim.Adam([
            {'params': actor_params, 'lr': actor_lr},
            {'params': critic_params, 'lr': critic_lr}
        ])
        
        # Old policy for computing probability ratios
        self.policy_old = ActorCritic().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()

    def update_learning_rate(self, epoch, base_actor_lr=None, base_critic_lr=None):
        """
        Update learning rate with schedule: lr * 32 / sqrt(1024 + epoch)
        """
        if base_actor_lr is None:
            base_actor_lr = self.actor_lr
        if base_critic_lr is None:
            base_critic_lr = self.critic_lr
            
        scale = 32.0 / math.sqrt(1024 + epoch)
        new_actor_lr = base_actor_lr * scale
        new_critic_lr = base_critic_lr * scale
        
        self.optimizer.param_groups[0]['lr'] = new_actor_lr
        self.optimizer.param_groups[1]['lr'] = new_critic_lr
        
        return new_actor_lr, new_critic_lr

    def preprocess_state(self, state):
        """
        Preprocess raw state to one-hot encoding.
        
        Args:
            state: numpy array (4, 4, 1) or (4, 4) or tensor
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Remove channel dim if present
        if state.dim() == 3 and state.size(-1) == 1:
            state = state.squeeze(-1)
        
        # Handle batch dimension
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        return one_hot_encode_fast(state)

    def select_action(self, state, deterministic=False):
        """
        Select action using current policy.
        
        Args:
            state: Raw game state (4, 4, 1) or (4, 4)
            deterministic: If True, select argmax action
            
        Returns:
            action: Selected action (0-3)
            log_prob: Log probability of action
        """
        with torch.no_grad():
            state_encoded = self.preprocess_state(state)
            logits, _ = self.policy_old(state_encoded)
            
            if deterministic:
                action = logits.argmax(dim=-1)
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(action)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item()

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        GAE(γ, λ) = Σ (γλ)^l * δ_{t+l}
        where δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        
        Args:
            rewards: list of rewards
            values: list of state values
            dones: list of done flags
            next_value: V(s_T) for bootstrapping
        
        Returns:
            returns: discounted returns
            advantages: GAE advantages
        """
        advantages = []
        gae = 0
        
        # Convert to numpy for computation
        values = np.array(values + [next_value])
        rewards = np.array(rewards)
        dones = np.array(dones, dtype=np.float32)
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values[:-1]
        
        return returns, advantages

    def normalize_advantages(self, advantages):
        """
        Normalize advantages using the formula from the report:
        A_norm = A / (3 * sqrt(mean(A^2)))
        A_reg = tanh(A_norm) * sqrt(|A_norm| + 0.692)
        """
        # Compute RMS normalization (assuming zero mean)
        rms = np.sqrt(np.mean(advantages ** 2) + 1e-8)
        adv_norm = advantages / (3 * rms)
        
        # Apply regularization transformation
        adv_reg = np.tanh(adv_norm) * np.sqrt(np.abs(adv_norm) + 0.692)
        
        return adv_reg

    def update(self, memory):
        """
        Update policy using collected experience.
        """
        # Get final value for GAE computation
        with torch.no_grad():
            if len(memory.states) > 0:
                last_state = memory.states[-1]
                last_state_encoded = self.preprocess_state(last_state.numpy())
                _, last_value = self.policy(last_state_encoded)
                next_value = last_value.item() if not memory.is_terminals[-1] else 0.0
            else:
                next_value = 0.0
        
        # Get values for all states
        with torch.no_grad():
            states_encoded = torch.cat([
                self.preprocess_state(s.numpy()) for s in memory.states
            ], dim=0)
            _, values = self.policy(states_encoded)
            values = values.squeeze(-1).cpu().numpy()
        
        # Compute GAE
        returns, advantages = self.compute_gae(
            memory.rewards, values.tolist(), memory.is_terminals, next_value
        )
        
        # Normalize advantages
        advantages = self.normalize_advantages(advantages)
        
        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_actions = torch.LongTensor(memory.actions).to(self.device)
        old_logprobs = torch.FloatTensor(memory.logprobs).to(self.device)
        
        # Preprocess all states
        old_states = torch.cat([
            self.preprocess_state(s.numpy()) for s in memory.states
        ], dim=0)

        # PPO update for K epochs
        for _ in range(self.k_epochs):
            # Get current policy outputs
            logits, values = self.policy(old_states)
            values = values.squeeze(-1)
            
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            
            # Probability ratio
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Policy loss (negative because we want to maximize)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = self.mse_loss(values, returns)
            
            # Entropy bonus (negative because we want to maximize entropy)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.critic_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }


class Memory:
    """Experience buffer for PPO."""
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
    
    def __len__(self):
        return len(self.states)
