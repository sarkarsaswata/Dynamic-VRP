from typing import Optional, Tuple, Dict, Any, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.stats import poisson
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import os
class DVRPEnv(gym.Env):
    """
    Custom Gym environment for the Discrete Vehicle Routing Problem (DVRP).

    Parameters:
        - env_size (int): Size of the environment grid.
        - task_arrival_rate (float): Rate at which tasks are generated per time step.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env_size=10, task_arrival_rate=0.8):
        super(DVRPEnv, self).__init__()

        # Environment parameters
        self.size = env_size
        self.task_arrival_rate = task_arrival_rate
        self.max_total_tasks = int(np.ceil(35 * self.task_arrival_rate))

        # Observation space limits
        self.obs_low  = np.array([0, 0] + [0, 0] * self.max_total_tasks + [0] * self.max_total_tasks, dtype=np.float32)
        self.obs_high = np.array([self.size, self.size] + [self.size, self.size] * self.max_total_tasks 
                            + [1500] * self.max_total_tasks, dtype=np.float32)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        # Intialize the environment
        # self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            # get a random seed
            seed = np.random.randint(0, 1000)
            np.random.seed(seed)
        # Reset episode-specific variables
        self.time_step = 0                  # episode time step
        self.new_task_period = 60           # set this number to generate new tasks after some interval
        self.task_step = 0                  # gets +=1 after new_task_period steps
        self.max_time_steps = 10            # set this number to draw discrete numbers from poisson
        self.total_tasks_serviced = 0       # the number of tasks serviced
        self.task_arrival_lst = poisson.rvs(mu=self.task_arrival_rate, size=self.max_time_steps) + 1
        
        # get total number of tasks in the environment
        self.total_tasks = np.sum(self.task_arrival_lst)

        # Initian state is the copy of self.obs_low
        self.state = self.obs_low.copy()

        # Initialize the UAV's position in the middle of the arena
        self.uav_pos = np.array([self.size // 2, self.size // 2], dtype=np.float32)
        self.depot = self.uav_pos.copy()
        self.state[:2] = self.uav_pos

        # create an array of task locations which are the slice of self.state from element 2 to 2 + 2 * self.max_total_tasks
        self.task_loc = self.state[2:2 + 2 * self.max_total_tasks]
        # use a helper function to generate new tasks
        self.task_loc = self.generate_new_tasks()

        # create an array of task times which are the slice of self.state from element 2 + 2 * self.max_total_tasks to the end
        self.times = self.state[2 + 2 * self.max_total_tasks:]

        info = {}

        # return the initial state
        return self.state, info

    def generate_new_tasks(self):
        # Check if there are more steps to generate tasks
        if self.task_step < self.max_time_steps:
            # get the number of new tasks to generate from task_arrival_lst with task_step as index
            self.task_arrival_lst = np.array(self.task_arrival_lst)
            num_new_tasks = self.task_arrival_lst[self.task_step]
            # if there are no new tasks, return the task locations
            if num_new_tasks == 0:
                return self.task_loc

            # otherwise, generate the new tasks
            else:
                # generate the new task locations
                new_tasks = np.random.randint(low=1, high=self.size, size=(num_new_tasks, 2)).astype(np.float32)
                new_tasks_reshaped = new_tasks.reshape(-1)
                new_tasks_reshaped = np.append(self.task_loc[self.task_loc > 0], new_tasks_reshaped)
                self.task_loc[:len(new_tasks_reshaped)] = new_tasks_reshaped
                return self.task_loc
            
        else:
            # No more steps to generate tasks
            return self.task_loc

    
    def step(self, action):
        # Ensure the action is within the action space
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f"Invalid action: {action}")
        
        self.time_step += 1

        self.uav_pos = self.move_agent(action)

        # Update waiting times for existing tasks
        self.update_waiting_times()

        task_loc_non_zero = self.task_loc[self.task_loc > 0].reshape(-1, 2)
        self.agent_at_task_location = any(
            np.array_equal(self.uav_pos, task_pos) for task_pos in task_loc_non_zero
        )

        if self.agent_at_task_location:
            self.total_tasks_serviced += 1
            self.task_loc, self.times = self.remove_completed_tasks()
        
        if self.time_step % self.new_task_period == 0:
            self.task_step += 1
            # Check if there are more steps in task_arrival_lst
            if self.task_step < self.max_time_steps:
                self.task_loc = self.generate_new_tasks()

        # Update state
        self.state = np.concatenate((self.uav_pos, self.task_loc, self.times), axis=None)

        # Terminated if and only if NO tasks are PRESENT and the agent has moved to the depot
        self.terminated = bool(np.all(self.task_loc == 0) and np.allclose(self.uav_pos, self.depot))

        self.truncated = False

        # Calculate reward
        reward = self.calculate_reward()

        # You can return additional information if needed
        info = {}

        return self.state, reward, self.terminated, self.truncated, info

    
    def move_agent(self, action):
        movements = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

        new_pos = self.uav_pos + np.array(movements[action], dtype=np.float32)
        new_pos = np.clip(new_pos, 0, self.size)
        return new_pos
    
    def update_waiting_times(self):
        task_loc_non_zero = self.task_loc[self.task_loc > 0].reshape(-1, 2)
        not_completed_tasks_mask = [not np.array_equal(task_pos, self.uav_pos) for task_pos in task_loc_non_zero]

        # Update waiting times only for tasks that are present in the environment
        self.times[:len(not_completed_tasks_mask)] += 1
    
    def remove_completed_tasks(self):
        tasks = self.task_loc[self.task_loc > 0]
        for i in range(0, len(tasks), 2):
            task_pos = tasks[i:i+2]
            if np.allclose(task_pos, self.uav_pos):
                self.task_loc[i:i+2] = 0
                self.times[i//2] = 0
        
        # Extract non-zero values from 'task_loc' and store them in 'non_zero_values'
        non_zero_values = self.task_loc[self.task_loc != 0]
        self.task_loc = np.zeros_like(self.task_loc)
        self.task_loc[:len(non_zero_values)] = non_zero_values

        non_zero_values = self.times[self.times != 0]
        self.times = np.zeros_like(self.times)
        self.times[:len(non_zero_values)] = non_zero_values

        return self.task_loc, self.times
    
    def calculate_reward(self):
        if self.agent_at_task_location:
            # Agent is at the same location as a task, reward is 0
            reward = 0
        
        else:
            if np.any(self.times > 0):
                # Agent is not at any task location, reward is negative of max time of the tasks
                reward = -np.max(self.times)
            
            else:
                # All tasks are completed, reward is the negative of the absolute distance from the depot
                reward = -np.abs(self.uav_pos - self.depot).sum()

        return float(reward)
    
    def render(self, mode="human", folder="render"):
        if mode == "human":
            self.game_over = self.terminated and self.truncated
            os.makedirs(folder, exist_ok=True)
        
        if not self.game_over:
            plt.figure(figsize=(8, 8))
            plt.scatter(self.uav_pos[0], self.uav_pos[1], marker="o", color="blue", s=100, label="UAV")
            plt.scatter(self.depot[0], self.depot[1], marker="*", color="green", s=100, label="Depot")

            task_locations = self.task_loc[self.task_loc > 0].reshape(-1, 2)
            plt.scatter(task_locations[:, 0], task_locations[:, 1], marker="s", color="red", s=100, label="Tasks")
            # Draw a bounding box around the arena
            plt.plot([0, self.size, self.size, 0, 0], [0, 0, self.size, self.size, 0], color='black', linewidth=2, linestyle='--')
            plt.title("DVRP Environment")
            plt.axis("off")
            plt.legend(loc="upper left")
            plt.xlim(0, self.size)
            plt.ylim(0, self.size)
            plt.savefig(os.path.join(folder, f"{self.time_step}.png"))    

if __name__ == "__main__":
    # Create environment instance
    env = DVRPEnv(task_arrival_rate=0.8)
    
    # Simulate over the environment for 150 steps
    obs = env.reset()
    
    # Run the simulation until done
    done = False
    step = 0
    total_reward = 0

    
    while not done and step < 10:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward

        done = terminated or truncated
        
        print("=^=" * 10)
        print(f"Step: {step + 1}")
        # print total number of tasks in the env
        print(f"Total tasks: {env.total_tasks}")
        # print total number of tasks serviced
        print(f"Total tasks serviced: {env.total_tasks_serviced}")
        # Print action
        print(f"Action: {action}")
        # Print observation
        print(f"Observation: {obs} of shape {obs.shape}")
        # Print reward
        print(f"Reward: {reward}")
        
        step += 1
    # print the task arrival list of env
    print(f"Task arrival list: {env.task_arrival_lst}")
    print(f"Total reward: {total_reward} after steps: {step}")
    # Check the environment
    check_env(env, warn=True)