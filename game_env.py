import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional

class Game2048Env(gym.Env):
    """
    Custom Gym environment for the 2048 game.
    
    The game is played on a 4x4 grid where tiles can be merged if they have the same value.
    The goal is to create a tile with the value 2048 (or higher).
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.grid_size = 4
        self.render_mode = render_mode
        
        # Action space: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 4x4 grid with tile values (log2 representation)
        # Values range from 0 (empty) to 17 (131072 = 2^17)
        self.observation_space = spaces.Box(
            low=0, high=17, shape=(4, 4), dtype=np.int32
        )
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the game to initial state."""
        super().reset(seed=seed)
        
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self.moves = 0
        self.invalid_moves = 0
        
        # Add two initial tiles
        self._add_random_tile()
        self._add_random_tile()
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=Up, 1=Right, 2=Down, 3=Left
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.moves += 1
        prev_score = self.score
        prev_board = self.board.copy()
        
        # Execute the move
        moved = self._move(action)
        
        # Calculate reward
        reward = 0
        if moved:
            # Base reward: log of score increase (compresses large values)
            merge_reward = self.score - prev_score
            if merge_reward > 0:
                reward += np.log2(merge_reward)  # log2(4)=2, log2(1024)=10
            
            # Bonus for creating higher tiles
            new_max = np.max(self.board)
            if new_max > self.max_tile:
                reward += new_max * 2
            self._add_random_tile()
            self.invalid_moves = 0
        else:
            # Penalty for invalid move
            reward = -2
            self.invalid_moves += 1
        
        # Update max tile
        self.max_tile = np.max(self.board)
        
        # Check if game is over (no valid moves OR too many consecutive invalid moves)
        terminated = not self._has_valid_moves() or self.invalid_moves >= 10
        truncated = False
        
        info = {
            'score': self.score,
            'max_tile': 2 ** self.max_tile if self.max_tile > 0 else 0,
            'moves': self.moves,
            'valid_move': moved
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Return the current board state."""
        return self.board.copy()
    
    def _add_random_tile(self) -> bool:
        """Add a random tile (2 or 4) to an empty cell."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        
        if not empty_cells:
            return False
        
        row, col = empty_cells[np.random.randint(len(empty_cells))]
        # 90% chance of 2, 10% chance of 4
        self.board[row, col] = 1 if np.random.random() < 0.9 else 2
        
        return True
    
    def _move(self, action: int) -> bool:
        """
        Execute a move in the specified direction.
        Returns True if the board changed, False otherwise.
        
        Actions: 0=Up, 1=Right, 2=Down, 3=Left
        0 (Up)    -> needs 1 rotation CCW (Top becomes Left)
        1 (Right) -> needs 2 rotations CCW (Right becomes Left)
        2 (Down)  -> needs 3 rotations CCW (Bottom becomes Left)
        3 (Left)  -> needs 0 rotations
        """
        rotation_map = {0: 1, 1: 2, 2: 3, 3: 0}
        k = rotation_map[action]

        rotated_board = np.rot90(self.board, k).copy()
        
        moved, new_board = self._move_left(rotated_board)
        
        if moved:
            # Rotate back to original orientation (-k)
            self.board = np.rot90(new_board, -k)
            
        return moved
    

    def _move_left(self, board: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Move all tiles left and merge. Returns (moved, new_board)."""
        moved = False
        
        for row in range(4):
            # Get non-zero tiles
            tiles = board[row][board[row] != 0]
            
            if len(tiles) == 0:
                continue
            
            # Merge tiles
            merged = []
            skip = False
            
            for i in range(len(tiles)):
                if skip:
                    skip = False
                    continue
                
                if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                    # Merge tiles
                    merged.append(tiles[i] + 1)
                    self.score += 2 ** (tiles[i] + 1)
                    skip = True
                    moved = True
                else:
                    merged.append(tiles[i])
            
            # Create new row
            new_row = merged + [0] * (4 - len(merged))
            
            if not np.array_equal(board[row], new_row):
                moved = True
            
            board[row] = new_row
        
        return moved, board
    
    def _has_valid_moves(self) -> bool:
        """Check if any valid moves are available."""
        # Check for empty cells
        if np.any(self.board == 0):
            return True
        
        # Check for possible merges horizontally
        for row in range(4):
            for col in range(3):
                if self.board[row, col] == self.board[row, col + 1]:
                    return True
        
        # Check for possible merges vertically
        for row in range(3):
            for col in range(4):
                if self.board[row, col] == self.board[row + 1, col]:
                    return True
        
        return False
    
    def render(self):
        """Render the current game state."""
        if self.render_mode == "human":
            print("\n" + "="*25)
            print(f"Score: {self.score} | Max Tile: {2**self.max_tile if self.max_tile > 0 else 0}")
            print("="*25)
            
            for row in self.board:
                tiles = [str(2**cell if cell > 0 else 0).rjust(5) for cell in row]
                print("|" + "|".join(tiles) + "|")
            
            print("="*25)
    
    def close(self):
        """Clean up resources."""
        pass
