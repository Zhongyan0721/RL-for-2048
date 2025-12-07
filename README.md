# 2048 Reinforcement Learning

This repository contains multiple Reinforcement Learning implementations for the 2048 game using different algorithms.

## Branches

| Branch | Algorithm | Description |
|--------|-----------|-------------|
| `main` | **PPO** | Proximal Policy Optimization |
| `dqn` | **DQN** | Deep Q-Network |
| `hdqn` | **HDQN** | Horizon Deep Q-Network |

## Installation

```bash
pip install -r requirements.txt
```

---

## PPO (main branch)

### Files
- `env_2048.py`: The 2048 game environment compatible with Gymnasium.
- `ppo.py`: Implementation of the PPO algorithm.
- `train.py`: Training script.

### Training
```bash
python train.py
```

The script will save the trained model to `ppo_2048_final.pth` and a plot of the training scores to `training_score.png`.

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
- `horizon_dqn.py`: Horizon DQN implementation.
- `train.py`: Training script.

### Training
```bash
python train.py
```

---

## Customization

You can adjust hyperparameters in `train.py` such as `max_episodes`, `lr`, `gamma`, etc.
You can also modify the network architecture or the reward function in the environment file.

