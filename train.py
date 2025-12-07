#!/usr/bin/env python
"""
Horizon DQN Training Script for 2048 Game

Usage:
    python train.py
"""

import os

import numpy as np
import torch
from tqdm import tqdm
import wandb

from game_env import Game2048Env
from horizon_dqn import HorizonAgent
from config import WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY


# ============== CONFIGURATION ==============
SEED = 42
NUM_EPISODES = 10000
SAVE_DIR = "checkpoints"

config = {
    "architecture": "Horizon DQN (Dueling + Noisy + QR-DQN + PER + N-step)",
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "n_step": 3,
    "num_atoms": 32,
    "buffer_size": 100000,
    "batch_size": 64,
    "target_update_freq": 1000,
    "num_episodes": NUM_EPISODES,
    "max_steps_per_episode": 10000,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0,
    "seed": SEED
}
# ===========================================


def setup_seeds(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_horizon_dqn(agent, env, config):
    """Train the Horizon DQN agent with beta annealing for PER."""
    num_episodes = config["num_episodes"]
    max_steps = config["max_steps_per_episode"]
    
    # Beta annealing for PER importance sampling
    beta_start = config["per_beta_start"]
    beta_end = config["per_beta_end"]
    
    # Best model tracking
    best_avg_score = 0
    best_max_tile = 0
    
    # Metrics for rolling average
    episode_scores = []
    
    for episode in tqdm(range(num_episodes), desc="Training Horizon DQN"):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        # Anneal beta linearly
        beta = beta_start + (beta_end - beta_start) * (episode / num_episodes)
        
        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.learn(beta=beta)
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_scores.append(info['score'])
        
        # Log to wandb
        metrics = {
            "episode": episode,
            "episode_reward": episode_reward,
            "episode_score": info['score'],
            "max_tile": info['max_tile'],
            "moves": info['moves'],
            "per_beta": beta,
            "buffer_size": len(agent.memory)
        }
        
        if episode_loss:
            metrics["avg_loss"] = np.mean(episode_loss)
        
        wandb.log(metrics)
        
        # Save best model based on rolling average score
        if episode >= 99:
            current_avg_score = np.mean(episode_scores[-100:])
            if current_avg_score > best_avg_score:
                best_avg_score = current_avg_score
                agent.save("horizon_dqn_best_score.pt", wandb_run_id=wandb.run.id)
        
        # Save model when new max tile is achieved
        if info['max_tile'] > best_max_tile:
            best_max_tile = info['max_tile']
            agent.save("horizon_dqn_best_max_tile.pt", wandb_run_id=wandb.run.id)
            print(f"\n New best tile: {best_max_tile}! Model saved.")
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(episode_scores[-100:]) if len(episode_scores) >= 100 else np.mean(episode_scores)
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Score (100): {avg_score:.2f}")
            print(f"  Best Max Tile: {best_max_tile}")
            print(f"  PER Beta: {beta:.4f}")
        
        # Save checkpoint every 1000 episodes
        if (episode + 1) % 1000 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"checkpoints/horizon_checkpoint_episode_{episode + 1}.pt")
            agent.save(checkpoint_path, wandb_run_id=wandb.run.id)
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
        name="horizon-dqn-2048",
        tags=["horizon-dqn", "rainbow-lite", "distributional", "2048"]
    )
    print(f"WandB run initialized: {run.name}")
    print(f"Run ID: {run.id}")
    print(f"View at: {run.url}")
    
    # Create environment and agent
    env = Game2048Env()
    agent = HorizonAgent(
        state_shape=(4, 4),
        num_actions=4,
        lr=config["learning_rate"],
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        target_update_freq=config["target_update_freq"],
        n_step=config["n_step"],
        num_atoms=config["num_atoms"]
    )
    
    print(f"Agent using device: {agent.device}")
    print(f"Total parameters: {sum(p.numel() for p in agent.online_net.parameters()):,}")
    
    # Train
    print(f"\nStarting Horizon DQN training for {NUM_EPISODES} episodes...")
    print("Note: No epsilon decay - Noisy Networks handle exploration!\n")
    train_horizon_dqn(agent, env, config)
    print("\nTraining complete!")
    
    # Save final model
    final_model_path = "horizon_dqn_final_model.pt"
    agent.save(final_model_path, wandb_run_id=wandb.run.id)
    print(f"Final model saved: {final_model_path}")
    
    # Finish WandB
    wandb.finish()
    print("WandB run finished!")


if __name__ == "__main__":
    main()
