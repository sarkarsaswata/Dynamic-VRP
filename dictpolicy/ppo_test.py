import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
from env import DVRPEnv
from maskaction import action_mask
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.utils import get_action_masks
from tqdm import tqdm
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                        default='ppo', help='Choose between ppo, a2c, or dqn')
    parser.add_argument('--size', type=int, 
                        default=10, help='Size of the grid (10, 20, 30, 40, 50)')
    parser.add_argument('--rate', type=float, 
                        default=0.5, help='Choose among [0.5, 0.6, 0.7, 0.8, 0.9]')
    parser.add_argument('--name', type=str, help='Name of the run')
    return parser.parse_args()

def test_model(args, num_ep=1):
    import json
    config_path = '/home/moonlab/MS_WORK/ms_thesis/dictpolicy/configs/config1.json'
    model_path = '/home/moonlab/MS_WORK/ms_thesis/dictpolicy/models/gnf59lwi.zip'

    with open(config_path, 'r') as f:
        params = json.load(f)
    
    env = DVRPEnv(**params)
    # env = ActionMasker(env, action_mask)
    # env = Monitor(env)

    # model = MaskablePPO.load(model_path)
    model = PPO.load(model_path)

    for ep in tqdm(range(num_ep)):
        actions = []
        rewards = []
        frames = []
        steps = 0
        obs, _ = env.reset()
        print(f"Initial state: {obs}")
        print("-"*50)
        done = False
        while not done:
            steps += 1

            # action_masks = get_action_masks(env)
            # action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            action, _ = model.predict(obs, deterministic=True)
            actions.append(int(action))

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            frame = env.render()
            frames.append(frame)

            print(f"Step {steps}")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print(f"State: {obs}")
            print("-"*50)

            rewards.append(reward)
            if done:
                env.close()
                # make video if render mode is 'rgb_array'
                if params["render_mode"] == "rgb_array":
                    from vid import create_video
                    create_video(frames, "gnf59lwi.mp4", fps=2)
                print(f"Episode {ep+1} completed in {steps} steps with reward {sum(rewards)}")
                print(f"Actions: {actions}")

if __name__ == '__main__':
    args = parse_args()
    test_model(args, num_ep=1)