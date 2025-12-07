#!/usr/bin/env python
"""
DQN Training Script for 2048 Game

Usage:
    python train.py
"""

import os

import numpy as np
import torch
from tqdm import tqdm
import wandb

from game_env import Game2048Env
from dqn_model import DQNAgent
from config import WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY


# ============== CONFIGURATION ==============
SEED = 42
SAVE_DIR = "checkpoints"

config = {
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "buffer_size": 100000,
    "batch_size": 64,
    "target_update_freq": 1000,
    "num_episodes": 10000,
    "max_steps_per_episode": 10000,
    "seed": SEED
}
# ===========================================


def setup_seeds(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_dqn(agent, env, config):
    """Train the DQN agent."""
    num_episodes = config["num_episodes"]
    max_steps = config["max_steps_per_episode"]
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        
        # Log to wandb
        metrics = {
            "episode": episode,
            "episode_reward": episode_reward,
            "episode_score": info['score'],
            "max_tile": info['max_tile'],
            "moves": info['moves'],
            "epsilon": agent.epsilon,
            "buffer_size": len(agent.replay_buffer)
        }
        
        if episode_loss:
            metrics["avg_loss"] = np.mean(episode_loss)
        
        wandb.log(metrics)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Score: {info['score']}, Max Tile: {info['max_tile']}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
        
        # Save checkpoint every 1000 episodes
        if (episode + 1) % 1000 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_episode_{episode + 1}.pt")
            agent.save(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")


def main():
    # Setup
    setup_seeds(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Initialize WandB
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=config,
        name="dqn-baseline",
        tags=["dqn", "baseline", "2048"]
    )
    print(f"WandB run initialized: {run.name}")
    print(f"Run ID: {run.id}")
    print(f"View at: {run.url}")
    
    # Create environment and agent
    env = Game2048Env()
    agent = DQNAgent(
        state_shape=(4, 4),
        num_actions=4,
        lr=config["learning_rate"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        target_update_freq=config["target_update_freq"]
    )
    
    print(f"Agent using device: {agent.device}")
    
    # Train
    print(f"\nStarting training for {NUM_EPISODES} episodes...\n")
    train_dqn(agent, env, config)
    print("\nTraining complete!")
    
    # Save final model
    final_model_path = "dqn_final_model.pt"
    agent.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Finish WandB
    wandb.finish()
    print("WandB run finished!")


if __name__ == "__main__":
    main()
