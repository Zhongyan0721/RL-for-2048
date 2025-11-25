import torch
import time
import numpy as np
from env_2048 import Game2048Env
from ppo import PPO

def play():
    env = Game2048Env(render_mode="human")
    agent = PPO(env)
    
    # Load trained model
    try:
        agent.policy.load_state_dict(torch.load('ppo_2048_final.pth'))
        print("Loaded trained model.")
    except:
        print("No trained model found, using random agent.")

    agent.policy.eval()
    
    state, _ = env.reset()
    done = False
    
    print("Initial State:")
    print(state[:, :, 0])

    while not done:
        action, _ = agent.select_action(state)
        state, reward, done, truncated, info = env.step(action)
        
        # Clear screen (optional, works in terminal)
        # print("\033[H\033[J") 
        print(f"\nAction: {['Up', 'Right', 'Down', 'Left'][action]}")
        print(state[:, :, 0])
        print(f"Score: {info['score']}")
        
        # time.sleep(0.5) # Slow down to watch

    print("Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Max Tile: {info['max_tile']}")

if __name__ == '__main__':
    play()

