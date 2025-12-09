# 2048 Reinforcement Learning

Deep Reinforcement Learning implementation for the 2048 game, based on the paper "Learning to Merge: Deep Reinforcement Learning Strategies for the 2048 Game".

## Features

### PPO Implementation (main branch)

- **Custom CNN Architecture**: Depthwise separable convolutions with directional kernels:
  - 1×4 horizontal kernels (row patterns)
  - 4×1 vertical kernels (column patterns)
  - 4×4 global kernels (full board)
- **One-Hot Encoding**: 16-channel input representation (one per tile value)
- **Generalized Advantage Estimation (GAE)**: λ = 0.9 for balanced bias-variance tradeoff
- **Potential-Based Reward Shaping**: Corner strategy with β = 64

### Hyperparameters (from paper)

| Parameter | Value |
|-----------|-------|
| Discount factor γ | 0.997 |
| GAE parameter λ | 0.9 |
| PPO clip ε | 0.1 |
| Actor learning rate | 2.5×10⁻⁵ |
| Critic learning rate | 6.25×10⁻⁵ |
| LR schedule | lr × 32 / √(1024 + epoch) |
| Batch size | 1024 |
| Entropy coefficient | 2.5×10⁻⁴ |
| Critic coefficient | ~10⁻⁸ |

## Branches

| Branch | Algorithm | Description |
|--------|-----------|-------------|
| `main` | **PPO** | Proximal Policy Optimization with custom CNN |
| `dqn` | **DQN** | Deep Q-Network baseline |
| `hdqn` | **HDQN** | Horizon DQN with distributional RL |

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Weights & Biases (optional, for DQN/HDQN branches):**
   
   Create a `config.py` file:
   ```python
   WANDB_API_KEY = "your-wandb-api-key-here"
   WANDB_PROJECT = "2048-rl"
   WANDB_ENTITY = "your-wandb-username"
   ```

## Files

- `env_2048.py`: 2048 game environment (Gymnasium compatible)
  - Logarithmic reward function
  - Potential-based corner shaping
  - Vectorized environment for parallel training
- `ppo.py`: PPO implementation
  - Custom CNN encoder with directional kernels
  - Actor-Critic architecture (1024→256→64 heads)
  - GAE with advantage normalization
- `train.py`: Training script with LR scheduling
- `play.py`: Evaluation and interactive play

---

## PPO (main branch)

### Training
```bash
python train.py
```

### Vectorized Training (faster)
```bash
python train.py --vectorized --num-envs 64
```

The script will:
- Save checkpoints every 1000 epochs
- Save final model to `ppo_2048_final.pth`
- Generate training plots to `training_results.png`

### Evaluation
```bash
python play.py --model ppo_2048_final.pth --games 100
```

### Interactive Play
```bash
python play.py --interactive
```
Controls: W=Up, D=Right, S=Down, A=Left, Q=Quit

---

## DQN (dqn branch)

```bash
git checkout dqn
```

### Files
- `game_env.py`: The 2048 game environment compatible with Gymnasium.
- `dqn_model.py`: DQN network and agent implementation.
- `train.py`: Training script with WandB logging.

### Training
```bash
python train.py
```

---

## HDQN (hdqn branch)

```bash
git checkout hdqn
```

### Files
- `game_env.py`: The 2048 game environment compatible with Gymnasium.
- `horizon_dqn.py`: Horizon DQN implementation with:
  - Distributional RL (QR-DQN)
  - Noisy Networks for exploration
  - Prioritized Experience Replay
  - Multi-step learning

### Training
```bash
python train.py
```

---

## Network Architecture

```
Input: One-hot encoded grid (16, 4, 4)
         ↓
┌────────┴────────┐
│   CNN Encoder   │
├─────┬─────┬─────┤
│1×4  │4×1  │4×4  │  Depthwise separable convs
│horiz│vert │glob │
└──┬──┴──┬──┴──┬──┘
   │     │     │
   └──┬──┴──┬──┘
      │concat│
      └──┬───┘
         ↓
   FC → 1024 features
         ↓
   ┌─────┴─────┐
   │           │
 Actor      Critic
1024→256   1024→256
 256→64     256→64
  64→4       64→1
   │           │
 logits     value
```

## Reward Function

```
R(s, a, s') = R_base + Φ(s') - Φ(s)

where:
  R_base = Σ log₂(merged_tile_values)
  Φ(s) = β × tile_value[0,0]  (corner potential)
  β = 64
```

## Results

The PPO agent achieves:
- **85%+ win rate** (reaching 2048 tile)
- Occasional games reaching **4096** and beyond
- Learned corner strategy through reward shaping

## Customization

Adjust hyperparameters in `train.py`:
- `max_epochs`: Number of training epochs
- `actor_lr`, `critic_lr`: Learning rates
- `gamma`, `gae_lambda`: Discount and GAE parameters
- `eps_clip`: PPO clipping parameter

Modify reward shaping in `env_2048.py`:
- `corner_bonus_factor`: β parameter (default 64)
- `invalid_penalty`: Penalty for invalid moves
- `reward_type`: 'raw', 'log', or 'shaped'
