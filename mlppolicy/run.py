import gym
import numpy as np
import os
import argparse
import wandb
from DVRPEnv import DVRPEnv
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model based on training reward, instead of training loss.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        
        return True
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                        default='ppo', help='Choose between ppo, a2c, or dqn')
    parser.add_argument('--size', type=int, 
                        default=10, help='Size of the grid (10, 20, 30, 40, 50)')
    parser.add_argument('--rate', type=float, 
                        default=0.5, help='Choose among [0.5, 0.6, 0.7, 0.8, 0.9]')
    return parser.parse_args()
    
def train_model(args):
    log_dir = f'./logs/{args.model}_logs'
    os.makedirs(log_dir, exist_ok=True)
    # Initialize wandb for logging
    wandb.init(project='DVRP_new', entity='persistent_routing', 
               name=f'{args.model}_lambda_{args.rate}', sync_tensorboard=True)

    # create environment instance
    env = DVRPEnv(env_size = args.size, 
                  task_arrival_rate = args.rate)
    
    env = Monitor(env, log_dir) # Wrap the environment with Monitor to log training progress

    # Choose the algotithm
    if args.model == 'ppo':
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f'./tblogs/{args.model}')
    elif args.model == 'a2c':
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=f'./tblogs/{args.model}')
    elif args.model == 'dqn':
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=f'./tblogs/{args.model}')
    else:
        raise ValueError('Invalid model choice. Choose between PPO, A2C, or DQN')
        
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=int(1e5), callback=callback)

    # close WandB
    wandb.finish()

    # save the trained model
    model_save_dir = os.path.join('saved_model', f'{args.model}_model')
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f'{args.model}_{args.rate}')
    model.save(model_path)

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # Check if task_arrival_rate is within the allowed choices
    if args.rate not in [0.5, 0.6, 0.7, 0.8, 0.9]:
        raise ValueError('Task arrival rate must be one of 0.5, 0.6, 0.7, 0.8, 0.9')
    
    # Train the model
    train_model(args)