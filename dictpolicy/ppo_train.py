import gymnasium as gym
import os, wandb, argparse
from callback import SaveOnBestTrainingRewardCallback
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from maskaction import action_mask
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib.common.wrappers import ActionMasker
from env import DVRPEnv
from stable_baselines3.common.monitor import Monitor

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

def train_model(args):
    log_dir = f'./logs/{args.model}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    wandb.init(project='DVRP_new',
               entity='persistent_routing', 
               name='ppo_unmasked',
               sync_tensorboard=True)
    
    # Get the WandB run ID so that we can use it to save the model
    # run_id = wandb.run.id

    # Number of parallel environments
    # num_envs = 4

    # create environment instance
    import json
    file_path = f'/home/moonlab/MS_WORK/ms_thesis/dictpolicy/configs/config1.json'
    with open(file_path, 'r') as f:
        params = json.load(f)
    # Create a function that returns an environment instance
    def make_env() -> Monitor:
        env = DVRPEnv(**params)
        env = ActionMasker(env, action_mask)
        env = Monitor(env, log_dir)
        return env
    
    # Create the vectorized environment
    # env = make_vec_env(make_env, n_envs=num_envs, monitor_dir=log_dir)
    env = make_env()
    # Choose the algotithm
    # model = PPO("MultiInputPolicy", 
    model = MaskablePPO("MultiInputPolicy", 
                env,
                n_steps=2000,
                batch_size=200,
                verbose=1,
                # device='cuda:0',
                tensorboard_log=f'./tblogs/{args.model}')
    
    callback = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir=log_dir)

    learn_kwargs = {'total_timesteps': int(50e3*200),   # 50k * 200 (episode length)  #run.id = eg25oner
    # learn_kwargs = {'total_timesteps': int(2*50e3*200),   # 100k * 200 (episode length)  
                    'callback': callback,
                    'reset_num_timesteps': False,
                    'progress_bar': True}
    
    model.learn(**learn_kwargs)
    
    env.close()
    wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    train_model(args)