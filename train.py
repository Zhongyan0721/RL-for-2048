import torch
import numpy as np
from env_2048 import Game2048Env
from ppo import PPO, Memory
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def train():
    env = Game2048Env()
    
    # --- Hyperparameters Tuning Section ---
    max_episodes = 100000   # Increase: 2048 needs lots of experience
    max_timesteps = 1000    # Max timesteps per episode
    update_timestep = 4000  # Increase: More data per update = more stable gradient
    lr = 1e-3               # Start higher, decay later
    lr_min = 1e-5           # Minimum learning rate
    gamma = 0.995
    epochs = 10             # Increase: Squeeze more out of each batch
    eps_clip = 0.2
    # --------------------------------------
    
    agent = PPO(env, lr, gamma, eps_clip, epochs)
    memory = Memory()
    
    timestep = 0
    
    # Logging
    log_interval = 50 # Log less frequently
    running_reward = 0
    running_score = 0
    running_max_tile = 0
    
    rewards_history = []
    scores_history = []
    
    print(f"Starting training for {max_episodes} episodes...")
    
    for i_episode in tqdm(range(1, max_episodes + 1)):
        state, _ = env.reset()
        current_ep_reward = 0
        
        # Linear Learning Rate Decay
        new_lr = lr - (lr - lr_min) * (i_episode / max_episodes)
        new_lr = max(new_lr, lr_min)
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = new_lr

        for t in range(max_timesteps):
            timestep += 1
            
            # Run old policy
            action, log_prob = agent.select_action(state)
            state_tensor = torch.FloatTensor(state)
            memory.states.append(state_tensor)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            
            state, reward, done, truncated, info = env.step(action)
            
            # Saving reward and is_terminals
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            current_ep_reward += reward
            
            # Update if its time
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0
            
            if done:
                break
        
        running_reward += current_ep_reward
        running_score += info['score']
        running_max_tile = max(running_max_tile, info['max_tile'])
        
        if i_episode % log_interval == 0:
            avg_reward = running_reward / log_interval
            avg_score = running_score / log_interval
            rewards_history.append(avg_reward)
            scores_history.append(avg_score)
            
            print(f'Episode {i_episode} \t Avg Reward: {avg_reward:.2f} \t Avg Score: {avg_score:.2f} \t Max Tile: {running_max_tile} \t LR: {new_lr:.6f}')
            running_reward = 0
            running_score = 0
            running_max_tile = 0
            
            # Save model
            if i_episode % 1000 == 0:
                torch.save(agent.policy.state_dict(), f'ppo_2048_{i_episode}.pth')

    # Save final model
    torch.save(agent.policy.state_dict(), 'ppo_2048_final.pth')
    
    # Plot results
    plt.figure()
    plt.plot(scores_history)
    plt.title('Average Score')
    plt.xlabel('Interval')
    plt.ylabel('Score')
    plt.savefig('training_score.png')
    print("Training finished. Model and plot saved.")

if __name__ == '__main__':
    train()
