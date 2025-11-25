# 2048 RL with PPO

This repository contains a Reinforcement Learning implementation for the 2048 game using Proximal Policy Optimization (PPO).

## Files

- `env_2048.py`: The 2048 game environment compatible with Gymnasium.
- `ppo.py`: Implementation of the PPO algorithm.
- `train.py`: Training script.
- `requirements.txt`: List of dependencies.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the agent, run:
```bash
python train.py
```

The script will save the trained model to `ppo_2048_final.pth` and a plot of the training scores to `training_score.png`.

## Customization

You can adjust hyperparameters in `train.py` such as `max_episodes`, `lr`, `gamma`, etc.
You can also modify the network architecture in `ppo.py` or the reward function in `env_2048.py`.

