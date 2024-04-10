import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np

def action_mask(env: gym.Env) -> np.ndarray:
    x, y = env.pos_agent    # type: ignore
    size = env.size        # type: ignore
    # Define the mapping from actions to movements
    # movements = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

    masks = np.ones(env.action_space.n, dtype=bool)   # type: ignore

    # If the agent is at the left edge, it can't move left or diagonally left
    if x == 0:
        masks[[1, 5, 7]] = False
    
    # If the agent is at the right edge, it can't move right or diagonally right
    if x == size - 1:
        masks[[0, 4, 5]] = False
    
    # If the agent is at the top edge, it can't move up or diagonally up
    if y == 0:
        masks[[3, 5, 7]] = False
    
    # If the agent is at the bottom edge, it can't move down or diagonally down
    if y == size - 1:
        masks[[2, 4, 6]] = False
    
    return masks