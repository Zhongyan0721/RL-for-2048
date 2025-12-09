import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math


class Game2048Env(gym.Env):
    """
    2048 Game Environment for Reinforcement Learning.
    
    State Space: 4x4 grid with tile values (0, 2, 4, 8, ..., 65536)
    Action Space: 4 discrete actions (Up, Right, Down, Left)
    
    Reward Function (from report):
    - Base reward: Sum of log2 values of merged tiles
    - Corner strategy shaping: Potential-based shaping with Φ(s) = β * 2^s[0]
    - Invalid move penalty: -α
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None, 
                 reward_type='shaped',
                 invalid_penalty=1.0,
                 corner_bonus_factor=64.0):
        """
        Initialize the 2048 environment.
        
        Args:
            render_mode: 'human' for terminal output, 'rgb_array' for image
            reward_type: 'raw' for game score, 'shaped' for potential-based shaping
            invalid_penalty: Penalty for invalid moves (α in the report)
            corner_bonus_factor: Scaling factor for corner shaping (β in the report, default 64)
        """
        super(Game2048Env, self).__init__()
        self.size = 4
        self.window_size = 512
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Reward configuration
        self.reward_type = reward_type
        self.invalid_penalty = invalid_penalty
        self.corner_bonus_factor = corner_bonus_factor

        # Actions: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_space = spaces.Discrete(4)
        
        # Observation: 4x4 grid
        self.observation_space = spaces.Box(
            low=0, high=2**16, shape=(self.size, self.size, 1), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self._add_new_tile()
        self._add_new_tile()
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Execute action and return (observation, reward, terminated, truncated, info).
        
        Reward function from report:
        - R_base(s, a, s') = Σ log2(v_m) for each merged tile value v_m
        - R_shaped = R_base + Φ(s') - Φ(s) where Φ(s) = β * 2^s[0]
        - R_invalid = -α for invalid moves
        """
        prev_grid = self.grid.copy()
        prev_corner_value = self.grid[0, 0]
        
        # Execute move
        moved = False
        merge_scores = []  # Track merged tile values for logarithmic reward
        
        if action == 0:  # Up
            moved, merge_scores = self._move_up()
        elif action == 1:  # Right
            moved, merge_scores = self._move_right()
        elif action == 2:  # Down
            moved, merge_scores = self._move_down()
        elif action == 3:  # Left
            moved, merge_scores = self._move_left()

        # Compute reward
        if not moved:
            # Invalid move penalty
            reward = -self.invalid_penalty
        else:
            # Add new tile after valid move
            self._add_new_tile()
            
            if self.reward_type == 'raw':
                # Raw reward: sum of merged tile values (game score)
                reward = float(sum(merge_scores))
            
            elif self.reward_type == 'log':
                # Logarithmic reward: sum of log2 of merged values
                reward = sum(math.log2(v) if v > 0 else 0 for v in merge_scores)
            
            else:  # 'shaped' (default)
                # Base reward: logarithmic
                base_reward = sum(math.log2(v) if v > 0 else 0 for v in merge_scores)
                
                # Potential-based shaping for corner strategy
                # Φ(s) = β * 2^rank(s[0]) where rank is log2 of tile value
                # This simplifies to Φ(s) = β * tile_value
                curr_corner_value = self.grid[0, 0]
                
                # Potential difference: Φ(s') - Φ(s)
                shaping_bonus = self.corner_bonus_factor * (curr_corner_value - prev_corner_value)
                
                reward = base_reward + shaping_bonus

        terminated = self._is_game_over()
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        info['moved'] = moved
        info['merge_scores'] = merge_scores

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """Return grid as observation with channel dimension."""
        return self.grid.astype(np.float32)[:, :, np.newaxis]

    def _get_info(self):
        """Return info dict with score and max tile."""
        return {
            "score": int(np.sum(self.grid)),
            "max_tile": int(np.max(self.grid)),
            "empty_cells": int(np.sum(self.grid == 0))
        }

    def _add_new_tile(self):
        """Add a new tile (2 or 4) to a random empty cell."""
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.grid[row, col] = 2 if random.random() < 0.9 else 4

    def _compress(self, row):
        """Compress row by moving all non-zero elements to the left."""
        new_row = [i for i in row if i != 0]
        new_row += [0] * (self.size - len(new_row))
        return new_row

    def _merge(self, row):
        """Merge adjacent equal tiles and return merged values."""
        merged_values = []
        for i in range(self.size - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                merged_values.append(row[i])  # Track the merged tile value
                row[i + 1] = 0
        return row, merged_values

    def _move_left_row(self, row):
        """Move and merge a single row to the left."""
        new_row = self._compress(row)
        new_row, merged_values = self._merge(new_row)
        new_row = self._compress(new_row)
        return np.array(new_row), merged_values

    def _move_left(self):
        """Move all tiles left and return (moved, merged_values)."""
        all_merged = []
        moved = False
        new_grid = np.zeros_like(self.grid)
        
        for i in range(self.size):
            new_row, merged_values = self._move_left_row(self.grid[i, :])
            all_merged.extend(merged_values)
            if not np.array_equal(self.grid[i, :], new_row):
                moved = True
            new_grid[i, :] = new_row
        
        self.grid = new_grid
        return moved, all_merged

    def _move_right(self):
        """Move all tiles right."""
        self.grid = np.fliplr(self.grid)
        moved, merged = self._move_left()
        self.grid = np.fliplr(self.grid)
        return moved, merged

    def _move_up(self):
        """Move all tiles up."""
        self.grid = self.grid.T
        moved, merged = self._move_left()
        self.grid = self.grid.T
        return moved, merged

    def _move_down(self):
        """Move all tiles down."""
        self.grid = self.grid.T
        moved, merged = self._move_right()
        self.grid = self.grid.T
        return moved, merged

    def _is_game_over(self):
        """Check if no valid moves remain."""
        # Check for empty cells
        if np.any(self.grid == 0):
            return False
        
        # Check for possible horizontal merges
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.grid[i, j] == self.grid[i, j + 1]:
                    return False
        
        # Check for possible vertical merges
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.grid[i, j] == self.grid[i + 1, j]:
                    return False
        
        return True

    def get_valid_actions(self):
        """Return mask of valid actions (actions that would change the board)."""
        valid = np.zeros(4, dtype=bool)
        original_grid = self.grid.copy()
        
        for action in range(4):
            self.grid = original_grid.copy()
            if action == 0:
                moved, _ = self._move_up()
            elif action == 1:
                moved, _ = self._move_right()
            elif action == 2:
                moved, _ = self._move_down()
            else:
                moved, _ = self._move_left()
            valid[action] = moved
        
        self.grid = original_grid
        return valid

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        """Render the game state."""
        if self.render_mode == "human":
            # Terminal rendering
            print("\n" + "=" * 25)
            for row in self.grid:
                print("|" + "|".join(f"{v:5d}" if v > 0 else "     " for v in row) + "|")
            print("=" * 25)
            print(f"Score: {np.sum(self.grid)}, Max: {np.max(self.grid)}")
        return None

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()


class VectorizedGame2048Env:
    """
    Vectorized 2048 environment for parallel training.
    
    Runs multiple games in parallel using NumPy operations.
    """
    
    def __init__(self, num_envs, reward_type='shaped', 
                 invalid_penalty=1.0, corner_bonus_factor=64.0):
        """
        Initialize vectorized environment.
        
        Args:
            num_envs: Number of parallel environments
            reward_type: 'raw', 'log', or 'shaped'
            invalid_penalty: Penalty for invalid moves
            corner_bonus_factor: β parameter for corner shaping
        """
        self.num_envs = num_envs
        self.size = 4
        self.reward_type = reward_type
        self.invalid_penalty = invalid_penalty
        self.corner_bonus_factor = corner_bonus_factor
        
        # Action and observation spaces (for compatibility)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=2**16, shape=(self.size, self.size, 1), dtype=np.float32
        )
        
        # Initialize all grids
        self.grids = np.zeros((num_envs, self.size, self.size), dtype=np.int32)
        
    def reset(self, seed=None):
        """Reset all environments."""
        if seed is not None:
            np.random.seed(seed)
        
        self.grids = np.zeros((self.num_envs, self.size, self.size), dtype=np.int32)
        
        # Add two tiles to each environment
        for env_idx in range(self.num_envs):
            self._add_tile(env_idx)
            self._add_tile(env_idx)
        
        return self._get_obs(), self._get_info()
    
    def reset_single(self, env_idx):
        """Reset a single environment."""
        self.grids[env_idx] = 0
        self._add_tile(env_idx)
        self._add_tile(env_idx)
    
    def _add_tile(self, env_idx):
        """Add a tile to a single environment."""
        empty = np.argwhere(self.grids[env_idx] == 0)
        if len(empty) > 0:
            idx = np.random.randint(len(empty))
            row, col = empty[idx]
            self.grids[env_idx, row, col] = 2 if np.random.random() < 0.9 else 4
    
    def _get_obs(self):
        """Get observations for all environments."""
        return self.grids.astype(np.float32)[:, :, :, np.newaxis]
    
    def _get_info(self):
        """Get info for all environments."""
        return {
            'scores': np.sum(self.grids, axis=(1, 2)),
            'max_tiles': np.max(self.grids, axis=(1, 2)),
            'empty_cells': np.sum(self.grids == 0, axis=(1, 2))
        }
    
    def step(self, actions):
        """
        Step all environments.
        
        Args:
            actions: Array of actions for each environment
            
        Returns:
            observations, rewards, dones, truncateds, infos
        """
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)
        
        prev_corners = self.grids[:, 0, 0].copy()
        
        for env_idx in range(self.num_envs):
            action = actions[env_idx]
            prev_grid = self.grids[env_idx].copy()
            
            # Execute move
            moved, merged = self._execute_move(env_idx, action)
            
            if not moved:
                rewards[env_idx] = -self.invalid_penalty
            else:
                self._add_tile(env_idx)
                
                if self.reward_type == 'raw':
                    rewards[env_idx] = sum(merged)
                elif self.reward_type == 'log':
                    rewards[env_idx] = sum(math.log2(v) if v > 0 else 0 for v in merged)
                else:  # shaped
                    base_reward = sum(math.log2(v) if v > 0 else 0 for v in merged)
                    shaping = self.corner_bonus_factor * (self.grids[env_idx, 0, 0] - prev_corners[env_idx])
                    rewards[env_idx] = base_reward + shaping
            
            dones[env_idx] = self._is_game_over(env_idx)
        
        return self._get_obs(), rewards, dones, np.zeros(self.num_envs, dtype=bool), self._get_info()
    
    def _execute_move(self, env_idx, action):
        """Execute a move for a single environment."""
        grid = self.grids[env_idx]
        prev_grid = grid.copy()
        
        if action == 0:  # Up
            grid = grid.T
            moved, merged = self._move_left_single(grid)
            self.grids[env_idx] = grid.T
        elif action == 1:  # Right
            grid = np.fliplr(grid)
            moved, merged = self._move_left_single(grid)
            self.grids[env_idx] = np.fliplr(grid)
        elif action == 2:  # Down
            grid = np.flipud(grid).T
            moved, merged = self._move_left_single(grid)
            self.grids[env_idx] = np.flipud(grid.T)
        else:  # Left
            moved, merged = self._move_left_single(grid)
            self.grids[env_idx] = grid
        
        return not np.array_equal(prev_grid, self.grids[env_idx]), merged
    
    def _move_left_single(self, grid):
        """Move a grid left (in-place) and return merged values."""
        merged = []
        for i in range(self.size):
            row = grid[i]
            # Compress
            non_zero = row[row != 0]
            # Merge
            j = 0
            while j < len(non_zero) - 1:
                if non_zero[j] == non_zero[j + 1]:
                    non_zero[j] *= 2
                    merged.append(non_zero[j])
                    non_zero = np.delete(non_zero, j + 1)
                j += 1
            # Fill back
            grid[i] = np.pad(non_zero, (0, self.size - len(non_zero)))
        return len(merged) > 0 or not np.array_equal(grid, grid), merged
    
    def _is_game_over(self, env_idx):
        """Check if game is over for a single environment."""
        grid = self.grids[env_idx]
        
        if np.any(grid == 0):
            return False
        
        # Check horizontal merges
        for i in range(self.size):
            for j in range(self.size - 1):
                if grid[i, j] == grid[i, j + 1]:
                    return False
        
        # Check vertical merges
        for j in range(self.size):
            for i in range(self.size - 1):
                if grid[i, j] == grid[i + 1, j]:
                    return False
        
        return True
