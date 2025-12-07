import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

class NoisyLinear(nn.Module):
    """
    NoisyLinear layer for Noisy Networks (Fortunato et al., 2018).
    Replaces epsilon-greedy exploration with learnable noise in the weights.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters: Mu (mean) and Sigma (std)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Sample new noise for the current step."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product for weight noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            # y = (mu + sigma * epsilon) * x
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # During evaluation, use only the mean
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


# -------------------------------------------------------------------------
# 2. Horizon-DQN Network (H-DQN)
#    Combines: Dueling + Noisy + Distributional (QR-DQN)
# -------------------------------------------------------------------------
class HorizonDQN(nn.Module):
    def __init__(self, input_shape=(4, 4), num_actions=4, num_atoms=32):
        super(HorizonDQN, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms # Quantiles per action
        
        # 1. Feature Extraction (CNN)
        # Using a 3-layer CNN as typical in high-performance 2048 agents
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0)
        
        # Calculating flatten size:
        # 4x4 -> 3x3 -> 2x2 -> 1x1
        # 128 * 1 * 1 = 128 (This CNN reduces board to a deep feature vector)
        self.flatten_dim = 128

        # 2. Dueling Architecture with Noisy Layers
        # Value Stream
        self.v_fc = NoisyLinear(self.flatten_dim, 256)
        self.v_head = NoisyLinear(256, num_atoms) # Outputs V distribution

        # Advantage Stream
        self.a_fc = NoisyLinear(self.flatten_dim, 256)
        self.a_head = NoisyLinear(256, num_actions * num_atoms) # Outputs A distribution

    def forward(self, x):
        # Input Preprocessing: Normalize log2 values
        # Max tile approx 2^16=65536, so divide by ~16-17
        x = x / 17.0 
        
        # Add channel dimension: (batch, 4, 4) -> (batch, 1, 4, 4)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:  # Single state without batch
            x = x.unsqueeze(0).unsqueeze(0)

        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # Value
        v = F.relu(self.v_fc(x))
        v = self.v_head(v) # (batch, num_atoms)
        v = v.view(-1, 1, self.num_atoms)

        # Advantage
        a = F.relu(self.a_fc(x))
        a = self.a_head(a) # (batch, actions * num_atoms)
        a = a.view(-1, self.num_actions, self.num_atoms)

        # Dueling Aggregation for Distributional RL
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s, .)))
        # Done on the atom/quantile level
        a_mean = a.mean(dim=1, keepdim=True)
        q_dist = v + (a - a_mean) # (batch, actions, atoms)

        return q_dist

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers."""
        self.v_fc.reset_noise()
        self.v_head.reset_noise()
        self.a_fc.reset_noise()
        self.a_head.reset_noise()


# -------------------------------------------------------------------------
# 3. SumTree for Prioritized Experience Replay (PER)
# -------------------------------------------------------------------------
class SumTree:
    """
    Binary Heap (Sum Tree) for efficient O(log N) sampling of priorities.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.count = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, n_step=3, gamma=0.99):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 0.01  # Small amount to prevent zero priority
        
        # N-step learning buffer
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        self.gamma = gamma

    def _get_n_step_info(self):
        """Compute n-step reward and next state."""
        reward, next_state, done = 0, None, False
        for i, transition in enumerate(self.n_step_buffer):
            r, n_s, d = transition[2], transition[3], transition[4]
            reward += (self.gamma ** i) * r
            if d:
                done = True
                next_state = n_s
                break
            else:
                next_state = n_s
        return reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        # Add to temporary n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute n-step transition
        r, ns, d = self._get_n_step_info()
        s, a = self.n_step_buffer[0][0], self.n_step_buffer[0][1]

        # Max priority for new data ensures it's replayed at least once
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
            
        self.tree.add(max_p, (s, a, r, ns, d))

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            # Handle potential empty data slots (edge case)
            if data == 0 or data is None or not isinstance(data, tuple):
                # Resample from beginning if invalid
                idx, p, data = self.tree.get(random.uniform(0, segment))
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(max(p, 1e-6))  # Prevent zero priority

        sampling_probabilities = np.array(priorities) / self.tree.total
        weights = (len(self.tree.data) * sampling_probabilities) ** (-beta)
        weights /= weights.max() # Normalize

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
            np.array(next_states), np.array(dones, dtype=np.float32),
            np.array(idxs), np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            
    def __len__(self):
        return self.tree.count


# -------------------------------------------------------------------------
# 4. H-DQN Agent
# -------------------------------------------------------------------------
class HorizonAgent:
    def __init__(self, state_shape=(4,4), num_actions=4, device=None,
                 lr=1e-4, gamma=0.99, buffer_size=100000, batch_size=64,
                 target_update_freq=1000, n_step=3, num_atoms=32):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.online_net = HorizonDQN(state_shape, num_actions, self.num_atoms).to(self.device)
        self.target_net = HorizonDQN(state_shape, num_actions, self.num_atoms).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr, eps=1e-4)
        
        # Memory (PER + N-step)
        self.memory = PrioritizedReplayBuffer(capacity=buffer_size, n_step=self.n_step, gamma=self.gamma)
        
        self.learn_step = 0
        self.learn_step_counter = 0  # Alias for compatibility
        
        # Quantile Regression Settings
        # Generate cumulative probabilities for quantiles (0.01 to 0.99)
        self.quantiles = torch.linspace(0.0, 1.0, self.num_atoms + 1).to(self.device)
        self.quantile_mid = (self.quantiles[:-1] + self.quantiles[1:]) / 2.0
        self.quantile_mid = self.quantile_mid.view(1, -1) # (1, num_atoms)
        
        # Noisy nets don't need epsilon, but add for compatibility
        self.epsilon = 0.0  # Exploration handled by noisy layers

    def select_action(self, state, training=True):
        # No epsilon-greedy needed; Noisy Nets handle exploration
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist = self.online_net(state_t) # (1, actions, atoms)
            q_values = dist.mean(dim=2) # (1, actions)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return None

        # 1. Sample from PER
        states, actions, rewards, next_states, dones, idxs, weights = self.memory.sample(self.batch_size, beta)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) # (batch, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device) # (batch, 1)
        weights = torch.FloatTensor(weights).to(self.device)

        # 2. Compute Distributional Loss (QR-DQN logic)
        
        # Current Quantiles: Theta(s, a)
        # Output: (batch, actions, atoms)
        dist_current = self.online_net(states) 
        # Select the atoms for the chosen action
        # Gather: (batch, atoms)
        dist_current = dist_current[range(self.batch_size), actions, :] 

        # Target Quantiles: r + gamma^n * Theta_target(s', a*)
        with torch.no_grad():
            # Double DQN for action selection
            next_dist_online = self.online_net(next_states)
            next_actions = next_dist_online.mean(dim=2).argmax(dim=1)
            
            next_dist_target = self.target_net(next_states)
            # Select atoms for best next action
            dist_next = next_dist_target[range(self.batch_size), next_actions, :]
            
            # Apply Bellman Update
            # T_theta = r + gamma^n * theta
            # Use n-step gamma
            gamma_n = self.gamma ** self.n_step
            target_quantiles = rewards + (1 - dones) * gamma_n * dist_next

        # 3. Quantile Huber Loss
        # Pairwise difference between current quantiles and target quantiles
        # diff: (batch, num_atoms_current, num_atoms_target)
        # We need to broadcast to compare every current atom with every target atom
        diff = target_quantiles.unsqueeze(1) - dist_current.unsqueeze(2)
        
        # Huber Loss (k=1 is standard)
        k = 1.0
        huber_loss = torch.where(diff.abs() <= k, 0.5 * diff.pow(2), k * (diff.abs() - 0.5 * k))
        
        # Quantile Weighting
        # tau shape: (1, num_atoms, 1) for broadcasting
        tau = self.quantile_mid.unsqueeze(2)
        # Standard QR Loss: |tau - I(error < 0)| * Huber(error)
        indicator = (diff < 0).float().detach()
        quantile_loss = torch.abs(tau - indicator) * huber_loss
        
        # Mean over BOTH atom dimensions (current and target), then per-sample loss
        # This gives a properly scaled loss value
        elementwise_loss = quantile_loss.mean(dim=(1, 2))  # (batch,)
        final_loss = (elementwise_loss * weights).mean()

        # 4. Optimize
        self.optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
        
        # 5. Update Priorities
        # Priority = abs(TD Error)
        # We use the elementwise loss as a proxy for TD error in distributional settings
        td_errors = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)
        
        # 6. Reset Noise
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # 7. Update Target Net
        self.learn_step += 1
        self.learn_step_counter = self.learn_step  # Alias
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return final_loss.item()
    
    def decay_epsilon(self):
        """No-op: Noisy Networks handle exploration, no epsilon needed."""
        pass
    
    def save(self, path, wandb_run_id=None):
        """Save checkpoint with optional WandB run ID for resume capability."""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learn_step': self.learn_step,
            'wandb_run_id': wandb_run_id  # Store for resuming WandB logging
        }, path)

    def load(self, path):
        """Load checkpoint. Returns wandb_run_id if available."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.learn_step = ckpt.get('learn_step', 0)
        self.learn_step_counter = self.learn_step
        return ckpt.get('wandb_run_id', None)