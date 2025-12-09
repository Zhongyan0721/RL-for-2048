import torch
import time
import numpy as np
from env_2048 import Game2048Env
from ppo import PPO, ActorCritic, one_hot_encode_fast
import argparse


def play(model_path=None, num_games=10, render=True, delay=0.0):
    """
    Play 2048 using a trained PPO agent.
    
    Args:
        model_path: Path to trained model weights
        num_games: Number of games to play
        render: Whether to render the game
        delay: Delay between moves (for visualization)
    """
    render_mode = "human" if render else None
    env = Game2048Env(render_mode=render_mode)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create policy network
    policy = ActorCritic().to(device)
    
    # Load trained weights
    if model_path:
        try:
            # Try loading full checkpoint first
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['policy_state_dict'])
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                policy.load_state_dict(checkpoint)
                print(f"Loaded model weights from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Using random policy.")
    else:
        print("No model specified, using random policy.")
    
    policy.eval()
    
    # Statistics
    scores = []
    max_tiles = []
    moves_per_game = []
    
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    for game in range(num_games):
        state, _ = env.reset()
        done = False
        num_moves = 0
        
        if render:
            print(f"\n{'='*40}")
            print(f"Game {game + 1}/{num_games}")
            print(f"{'='*40}")
        
        while not done:
            # Preprocess state
            grid = state.squeeze(-1) if state.ndim == 3 else state
            grid_tensor = torch.FloatTensor(grid).unsqueeze(0).to(device)
            one_hot = one_hot_encode_fast(grid_tensor)
            
            # Get action
            with torch.no_grad():
                logits, value = policy(one_hot)
                action = logits.argmax(dim=-1).item()
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            num_moves += 1
            
            if render:
                print(f"\nMove {num_moves}: {action_names[action]}")
                if delay > 0:
                    time.sleep(delay)
            
            state = next_state
        
        # Record statistics
        scores.append(info['score'])
        max_tiles.append(info['max_tile'])
        moves_per_game.append(num_moves)
        
        if render:
            print(f"\n--- Game Over ---")
            print(f"Score: {info['score']}")
            print(f"Max Tile: {info['max_tile']}")
            print(f"Moves: {num_moves}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Games played: {num_games}")
    print(f"Average score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"Max score: {max(scores)}")
    print(f"Average moves: {np.mean(moves_per_game):.1f}")
    print(f"\nMax tile distribution:")
    
    # Count tile occurrences
    unique_tiles, counts = np.unique(max_tiles, return_counts=True)
    for tile, count in sorted(zip(unique_tiles, counts), reverse=True):
        pct = 100 * count / num_games
        print(f"  {tile:5d}: {count:3d} ({pct:5.1f}%)")
    
    # Win rate (reaching 2048+)
    win_count = sum(1 for t in max_tiles if t >= 2048)
    print(f"\nWin rate (2048+): {100 * win_count / num_games:.1f}%")
    
    return scores, max_tiles


def interactive_play():
    """
    Play 2048 interactively with keyboard controls.
    """
    env = Game2048Env(render_mode="human")
    state, _ = env.reset()
    
    print("\nInteractive 2048!")
    print("Controls: W=Up, D=Right, S=Down, A=Left, Q=Quit")
    print("=" * 40)
    
    action_map = {'w': 0, 'd': 1, 's': 2, 'a': 3}
    done = False
    
    while not done:
        key = input("\nYour move: ").lower().strip()
        
        if key == 'q':
            print("Quitting...")
            break
        
        if key not in action_map:
            print("Invalid key. Use W/A/S/D to move.")
            continue
        
        action = action_map[key]
        state, reward, done, truncated, info = env.step(action)
        
        if not info.get('moved', True):
            print("Invalid move! That direction doesn't change the board.")
    
    print(f"\nFinal Score: {info['score']}")
    print(f"Max Tile: {info['max_tile']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play 2048 with trained agent')
    parser.add_argument('--model', '-m', type=str, default='ppo_2048_final.pth',
                        help='Path to trained model')
    parser.add_argument('--games', '-g', type=int, default=10,
                        help='Number of games to play')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--delay', '-d', type=float, default=0.0,
                        help='Delay between moves (seconds)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Play interactively with keyboard')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_play()
    else:
        play(
            model_path=args.model,
            num_games=args.games,
            render=not args.no_render,
            delay=args.delay
        )
