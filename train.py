import torch
import numpy as np
from env_2048 import Game2048Env
from ppo import PPO, Memory
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import math


def train():
    """
    PPO training for 2048 with hyperparameters from the report.
    
    Hyperparameters (Table 1):
    - Discount factor γ = 0.997
    - GAE parameter λ = 0.9  
    - PPO clip parameter ε = 0.1
    - Base actor learning rate = 2.5×10^-5
    - Base critic learning rate = 6.25×10^-5
    - Learning rate schedule: lr × 32 / √(1024 + epoch)
    - Batch size = 1024
    - Entropy coefficient = 2.5×10^-4
    - Critic loss coefficient ≈ 10^-8
    - Parallel environments = 4096 (simplified to single env here)
    - Steps per epoch = 16
    - Transition reuse count = 2
    """
    env = Game2048Env()
    
    # --- Hyperparameters from Report ---
    max_epochs = 50000           # Number of training epochs
    steps_per_epoch = 16         # Steps per epoch (from report)
    batch_size = 1024            # Batch size for updates
    
    # Learning rates
    actor_lr = 2.5e-5
    critic_lr = 6.25e-5
    
    # PPO hyperparameters
    gamma = 0.997                # Discount factor
    gae_lambda = 0.9             # GAE parameter
    eps_clip = 0.1               # PPO clip parameter
    k_epochs = 4                 # Number of PPO update epochs
    
    # Loss coefficients
    entropy_coef = 2.5e-4        # Entropy coefficient
    critic_coef = 1e-8           # Critic loss coefficient
    
    # Update frequency
    update_timestep = steps_per_epoch * 64  # Collect ~1024 transitions before update
    # ----------------------------------
    
    agent = PPO(
        env,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        eps_clip=eps_clip,
        k_epochs=k_epochs,
        entropy_coef=entropy_coef,
        critic_coef=critic_coef
    )
    memory = Memory()
    
    timestep = 0
    episode = 0
    epoch = 0
    
    # Logging
    log_interval = 100
    save_interval = 1000
    
    running_reward = 0
    running_score = 0
    running_max_tile = 0
    episode_count = 0
    
    rewards_history = []
    scores_history = []
    max_tiles_history = []
    loss_history = []
    
    print(f"Starting PPO training for {max_epochs} epochs...")
    print(f"Device: {agent.device}")
    print(f"Actor LR: {actor_lr}, Critic LR: {critic_lr}")
    print(f"Gamma: {gamma}, GAE Lambda: {gae_lambda}, Clip: {eps_clip}")
    print("-" * 60)
    
    state, _ = env.reset()
    current_ep_reward = 0
    
    pbar = tqdm(total=max_epochs, desc="Training")
    
    while epoch < max_epochs:
        # Update learning rate based on epoch
        current_actor_lr, current_critic_lr = agent.update_learning_rate(epoch)
        
        # Collect experience
        for _ in range(steps_per_epoch):
            timestep += 1
            
            # Select action
            action, log_prob = agent.select_action(state)
            
            # Store transition
            state_tensor = torch.FloatTensor(state.squeeze(-1) if state.ndim == 3 else state)
            memory.states.append(state_tensor)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            current_ep_reward += reward
            state = next_state
            
            if done:
                # Log episode stats
                running_reward += current_ep_reward
                running_score += info['score']
                running_max_tile = max(running_max_tile, info['max_tile'])
                episode_count += 1
                episode += 1
                
                # Reset
                state, _ = env.reset()
                current_ep_reward = 0
        
        # Update policy at end of epoch
        if len(memory) >= update_timestep or len(memory) >= batch_size:
            loss_info = agent.update(memory)
            memory.clear_memory()
            loss_history.append(loss_info)
        
        epoch += 1
        pbar.update(1)
        
        # Logging
        if epoch % log_interval == 0 and episode_count > 0:
            avg_reward = running_reward / episode_count
            avg_score = running_score / episode_count
            
            rewards_history.append(avg_reward)
            scores_history.append(avg_score)
            max_tiles_history.append(running_max_tile)
            
            pbar.set_postfix({
                'Epoch': epoch,
                'Episodes': episode,
                'Avg Score': f'{avg_score:.0f}',
                'Max Tile': running_max_tile,
                'LR': f'{current_actor_lr:.2e}'
            })
            
            # Reset running stats
            running_reward = 0
            running_score = 0
            running_max_tile = 0
            episode_count = 0
        
        # Save checkpoint
        if epoch % save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'episode': episode,
                'policy_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'rewards_history': rewards_history,
                'scores_history': scores_history,
            }
            torch.save(checkpoint, f'ppo_2048_epoch_{epoch}.pth')
            torch.save(agent.policy.state_dict(), f'ppo_2048_{epoch}.pth')
    
    pbar.close()
    
    # Save final model
    torch.save(agent.policy.state_dict(), 'ppo_2048_final.pth')
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Score history
    axes[0, 0].plot(scores_history)
    axes[0, 0].set_title('Average Score per Interval')
    axes[0, 0].set_xlabel('Interval')
    axes[0, 0].set_ylabel('Score')
    
    # Reward history
    axes[0, 1].plot(rewards_history)
    axes[0, 1].set_title('Average Reward per Interval')
    axes[0, 1].set_xlabel('Interval')
    axes[0, 1].set_ylabel('Reward')
    
    # Max tile history
    if max_tiles_history:
        axes[1, 0].plot(max_tiles_history)
        axes[1, 0].set_title('Max Tile per Interval')
        axes[1, 0].set_xlabel('Interval')
        axes[1, 0].set_ylabel('Max Tile')
        axes[1, 0].set_yscale('log', base=2)
    
    # Loss history
    if loss_history:
        policy_losses = [l['policy_loss'] for l in loss_history]
        axes[1, 1].plot(policy_losses)
        axes[1, 1].set_title('Policy Loss')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.close()
    
    print("\n" + "=" * 60)
    print("Training finished!")
    print(f"Total epochs: {epoch}")
    print(f"Total episodes: {episode}")
    print(f"Model saved to: ppo_2048_final.pth")
    print(f"Training plot saved to: training_results.png")
    print("=" * 60)


def train_vectorized(num_envs=64):
    """
    Vectorized training with multiple parallel environments.
    
    This is closer to the report's setup with 4096 parallel envs,
    but scaled down for practical use.
    """
    from env_2048 import Game2048Env
    
    # Create vectorized environments
    envs = [Game2048Env() for _ in range(num_envs)]
    
    # Hyperparameters from report
    max_epochs = 50000
    steps_per_epoch = 16
    
    actor_lr = 2.5e-5
    critic_lr = 6.25e-5
    gamma = 0.997
    gae_lambda = 0.9
    eps_clip = 0.1
    k_epochs = 4
    entropy_coef = 2.5e-4
    critic_coef = 1e-8
    
    # Initialize agent (use first env for reference)
    agent = PPO(
        envs[0],
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        eps_clip=eps_clip,
        k_epochs=k_epochs,
        entropy_coef=entropy_coef,
        critic_coef=critic_coef
    )
    
    # Initialize states for all environments
    states = []
    for env in envs:
        state, _ = env.reset()
        states.append(state)
    states = np.stack(states)
    
    memory = Memory()
    
    epoch = 0
    total_episodes = 0
    running_scores = []
    running_max_tiles = []
    
    print(f"Starting vectorized PPO training...")
    print(f"Number of parallel environments: {num_envs}")
    print(f"Transitions per epoch: {num_envs * steps_per_epoch}")
    
    pbar = tqdm(total=max_epochs, desc="Training")
    
    while epoch < max_epochs:
        # Update learning rate
        agent.update_learning_rate(epoch)
        
        # Collect experience from all environments
        for step in range(steps_per_epoch):
            # Batch select actions
            actions = []
            log_probs = []
            
            for i, state in enumerate(states):
                action, log_prob = agent.select_action(state)
                actions.append(action)
                log_probs.append(log_prob)
                
                # Store transition
                state_tensor = torch.FloatTensor(state.squeeze(-1) if state.ndim == 3 else state)
                memory.states.append(state_tensor)
                memory.actions.append(action)
                memory.logprobs.append(log_prob)
            
            # Step all environments
            next_states = []
            for i, (env, action) in enumerate(zip(envs, actions)):
                next_state, reward, done, truncated, info = env.step(action)
                
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                if done:
                    running_scores.append(info['score'])
                    running_max_tiles.append(info['max_tile'])
                    total_episodes += 1
                    next_state, _ = env.reset()
                
                next_states.append(next_state)
            
            states = np.stack(next_states)
        
        # Update policy
        if len(memory) >= 512:  # Minimum batch size
            agent.update(memory)
            memory.clear_memory()
        
        epoch += 1
        pbar.update(1)
        
        # Logging
        if epoch % 100 == 0 and running_scores:
            avg_score = np.mean(running_scores[-100:])
            max_tile = max(running_max_tiles[-100:]) if running_max_tiles else 0
            
            pbar.set_postfix({
                'Epoch': epoch,
                'Episodes': total_episodes,
                'Avg Score': f'{avg_score:.0f}',
                'Max Tile': max_tile
            })
        
        # Save checkpoint
        if epoch % 1000 == 0:
            torch.save(agent.policy.state_dict(), f'ppo_2048_vec_{epoch}.pth')
    
    pbar.close()
    
    # Save final model
    torch.save(agent.policy.state_dict(), 'ppo_2048_vec_final.pth')
    print(f"Training complete! Total episodes: {total_episodes}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent for 2048')
    parser.add_argument('--vectorized', '-v', action='store_true',
                        help='Use vectorized training with multiple environments')
    parser.add_argument('--num-envs', '-n', type=int, default=64,
                        help='Number of parallel environments for vectorized training')
    args = parser.parse_args()
    
    if args.vectorized:
        train_vectorized(num_envs=args.num_envs)
    else:
        train()
