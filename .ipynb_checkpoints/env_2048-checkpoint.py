import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Game2048Env(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super(Game2048Env, self).__init__()
        self.size = 4
        self.window_size = 512
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Actions: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_space = spaces.Discrete(4)
        
        # Observation: 4x4 grid, values are powers of 2. 
        # We can use log2 values to keep numbers small for NN or just raw values.
        # Using log2 representation might be better for stability, but let's stick to raw or normalized.
        # Max tile is usually 2048 (2^11) or higher (2^16=65536).
        # Let's use a Box space.
        self.observation_space = spaces.Box(low=0, high=2**16, shape=(self.size, self.size, 1), dtype=np.float32)

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
        # print(self.grid)
        # print(self.grid[0, 0])
        # 0: Up, 1: Right, 2: Down, 3: Left
        
        prev_grid = self.grid.copy()
        score_gain = 0
        moved = False

        if action == 0: # Up
            moved, score_gain = self._move_up()
        elif action == 1: # Right
            moved, score_gain = self._move_right()
        elif action == 2: # Down
            moved, score_gain = self._move_down()
        elif action == 3: # Left
            moved, score_gain = self._move_left()

        # Reward Engineering
        reward = 0.0
        
        if not moved:
            reward = -10.0
        else:
            self._add_new_tile()
            
            # Reward 1
            if score_gain > 0:
                reward += score_gain

            # Reward 2
            extra = 0
            factor = 64
            if prev_grid[0, 0] != 0:
                extra -= factor * prev_grid[0, 0]
            if self.grid[0, 0] != 0:
                extra += factor * self.grid[0, 0]
            reward += extra
            

        terminated = self._is_game_over()
        truncated = False # Infinite horizon until game over
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # Return grid as observation
        # Add a channel dimension
        return self.grid.astype(np.float32)[:, :, np.newaxis]

    def _get_info(self):
        return {"score": np.sum(self.grid), "max_tile": np.max(self.grid)}

    def _add_new_tile(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.grid[row, col] = 2 if random.random() < 0.9 else 4

    def _compress(self, row):
        new_row = [i for i in row if i != 0]
        new_row += [0] * (self.size - len(new_row))
        return new_row

    def _merge(self, row):
        score = 0
        for i in range(self.size - 1):
            if row[i] != 0 and row[i] == row[i+1]:
                row[i] *= 2
                score += row[i]
                row[i+1] = 0
        return row, score

    def _move_left_row(self, row):
        new_row = self._compress(row)
        new_row, score = self._merge(new_row)
        new_row = self._compress(new_row)
        return np.array(new_row), score

    def _move_left(self):
        score = 0
        moved = False
        new_grid = np.zeros_like(self.grid)
        for i in range(self.size):
            new_row, row_score = self._move_left_row(self.grid[i, :])
            score += row_score
            if not np.array_equal(self.grid[i, :], new_row):
                moved = True
            new_grid[i, :] = new_row
        self.grid = new_grid
        return moved, score

    def _move_right(self):
        # Reverse, move left, reverse back
        self.grid = np.fliplr(self.grid)
        moved, score = self._move_left()
        self.grid = np.fliplr(self.grid)
        return moved, score

    def _move_up(self):
        # Transpose, move left, transpose back
        self.grid = self.grid.T
        moved, score = self._move_left()
        self.grid = self.grid.T
        return moved, score

    def _move_down(self):
        # Transpose, move right, transpose back
        self.grid = self.grid.T
        moved, score = self._move_right()
        self.grid = self.grid.T
        return moved, score

    def _is_game_over(self):
        if np.any(self.grid == 0):
            return False
        # Check possible merges
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.grid[i, j] == self.grid[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.grid[i, j] == self.grid[i+1, j]:
                    return False
        return True

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Simplified rendering (just print grid for now if human, return None)
        if self.render_mode == "human":
            # print("\n" + str(self.grid))
            pass
        return None

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

