"""
This file, env.py, implements the environment for the DVRP problem using the OpenAI Gym API.
"""

from typing import Tuple, Union
import gymnasium as gym
import gymnasium.spaces as spaces
import gymnasium.utils.seeding as seeding
import numpy as np
import seaborn as sns
import gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from decimal import Decimal
from gymnasium.wrappers.flatten_observation import FlattenObservation
from stable_baselines3.common.env_checker import check_env
from dvrp.components.task import Task, TaskStatus
from dvrp.components.gen_tasks import PoissonTaskGenerator


class DVRPEnv(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array"],
                'render_fps': 4
                }
    def __init__(self, size=10, lam=0.5, n_tasks=10, max_omega = 20, global_max_step=200, render_mode=None):
        """
        Initialize the DVRPEnv class.

        Parameters:
        - size (int): The size of the environment.
        - lam (float): The task arrival rate (lambda) for the Poisson task generator.
        - n_tasks (int): The maximum number of tasks to generate in the episode that will be used to train the agent.
        - max_omega (int): The maximum angular velocity of the agent.
        - global_max_step (int): The maximum time in the environment.
        - render_mode (str or None): The mode to use for rendering the environment. If None, the environment will not be rendered.

        Returns:
        None
        """
        super(DVRPEnv, self).__init__()
        assert 0.5 <= lam <= 0.9, "lambda must be in the range [0.5, 0.9]"
        self.size = size
        self.rate = lam
        self.max_tasks = n_tasks
        self.MAX_ANGULAR_VELOCITY = max_omega * np.pi / 180
        self.max_time = global_max_step         # The maximum time for the tasks
        self.total_time = self.max_time + 50    # The total time for the environment episode #250

        # action space: angular velocity of the agent; normalized to [-1, 1]
        action_low = np.array([-1])
        action_high = np.array([1])
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(1, ), dtype=np.float32)

        # observation space
        self.observation_space = spaces.Dict(
            {
                "position_agent" : spaces.Box(low=0, high=self.size, shape=(2, ), dtype=np.float32),
                "position_task" : spaces.Box(low=0, high=self.size, shape=(self.max_tasks*2, ), dtype=np.float32),
                "time_clock" : spaces.Box(low=0, high=self.max_time*2, shape=(self.max_tasks, ), dtype=np.float32),
                "z" : spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32)
            }
        )

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.task_colors = dict()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int): Seed for the random number generator. If None, a random seed will be used.
            options (dict): Additional options for resetting the environment. Default set to None.

        Returns:
            obs (dict): The initial observation of the environment.
            info (dict): Additional information about the environment's state.
        """
        _, seed = seeding.np_random(seed)
        super().reset(seed=seed, options=options)
        self.time = 0           # Time in seconds 
        self.steps = 0          # The number of steps/action taken by the agent
        self.v = 1              # velocity of the agent to 1
        self.yaw = 0            # angular velocity of the agent to 0
        self.dt = 0.1           # delta time 0..1s for executing the action
        self.tasks_done = 0     # number of tasks serviced by the agent

        # Agent's initial position is the middle of the environment
        self.agent_pos = np.array([self.size // 2, self.size // 2], dtype=np.float32)

        # generate tasks using the poisson task generator and store the arrival times in self.arrival_times list
        self.tasks_list = PoissonTaskGenerator(total_tasks=self.max_tasks, max_time=self.max_time, seed=seed).get_tasks(lam=self.rate)
        self.arrival_times = np.array([task.time_created for task in self.tasks_list], dtype=np.float32).reshape(-1)

        # Initialize the tasks, clocks, and progress arrays
        self.tasks_pos = self._get_tasks(time=self.time)
        self.clocks = self._get_clocks()
        self.progress = self._get_progress(time=self.time)
        
        self.agent_on_task = self._is_agent_on_task()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info
    
    def step(self, action):
        """
        Perform a step in the environment.

        Args:
            action: The action to be taken by the agent.

        Returns:
            obs (object): The observation of the environment after the step.
            reward (float): The reward obtained from the step.
            terminated (bool): Whether the episode is terminated after the step.
            truncated (bool): Whether the episode is truncated after the step.
            info (dict): Additional information about the step.

        """
        # increase the time step with self.dt and round the value to one decimal place
        self.time = np.round(self.time + self.dt, 1)
        self.steps += 1             # number of steps/actions taken by the agent
        self.agent_pos = self._move_agent(action)
        self.clocks = self._update_clocks()
        self.agent_on_task = self._is_agent_on_task()

        if self.agent_on_task:
            # if the agent is on a task, then service the task
            self.tasks_pos, self.clocks = self._remove_task()
            self.tasks_done = sum(task.status == TaskStatus.SERVICED for task in self.tasks_list)

        self.tasks_pos = self._get_tasks(time=self.time)
        self.progress = self._get_progress(time=self.time)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        if terminated or truncated:
            # get the list of tasks that are pending
            pending_tasks = [task for task in self.tasks_list if task.status == TaskStatus.PENDING and task.time_created <= self.time]
            # mark the pending tasks as incomplete
            for task in pending_tasks:
                task.incomplete(self.time)
            
        reward = self._get_reward(term=terminated, trunc=truncated)
        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def close(self):
        pass
    
    def _get_tasks(self, time: float) -> np.ndarray:
        """
        Get the locations of tasks at a given time. Initiated in the reset method.

        Parameters:
        - time (float): Current time.

        Returns:
        - tasks_locs (numpy.ndarray): An array containing the locations of tasks at the given time.
        """
        if hasattr(self, 'tasks_pos'):
            tasks_locs = np.copy(self.tasks_pos)
        else:
            tasks_locs = np.zeros((self.max_tasks*2, ), dtype=np.float32)

        matching_indices = np.where(self.arrival_times == time)[0]
        if matching_indices.size == 0:
            return tasks_locs

        for index in matching_indices:
            task = self.tasks_list[index]
            x_index = index * 2
            y_index = x_index + 1
            tasks_locs[x_index] = task.location[0]
            tasks_locs[y_index] = task.location[1]
        return tasks_locs
    
    def _get_clocks(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Array of clocks.
        """
        return np.zeros((self.max_tasks, ), dtype=np.float32)
    
    def _get_progress(self, time: float) -> np.ndarray:
        """
        Calculate the progress based on the given time. Initiated in the reset method.

        Parameters:
        time (float): Current time.

        Returns:
        np.ndarray: An array containing the calculated progress.

        Raises:
        ValueError: If the calculated progress is not within the range [-1, 1].
        """
        min_progress, max_progress = -1.0, 1.0
        progress = float(time / self.total_time)
        progress = min_progress + progress * (max_progress - min_progress)
        if not min_progress <= progress <= max_progress:
            raise ValueError(f"Progress must be in the range [-1, 1], but got {progress}")
        return np.array([progress], dtype=np.float32)
    
    def _is_agent_on_task(self) -> bool:
        """
        Checks if the agent is on a task location and services the pending tasks. Initiated in the reset method.

        Returns:
            bool: True if there are tasks serviced, False otherwise.
        """
        tasks_serviced = False
        agent_pos = np.copy(self.agent_pos)

        task_statuses = np.array([task.status for task in self.tasks_list], dtype=bool)
        task_times = np.array([task.time_created for task in self.tasks_list], dtype=np.float32)

        task_is_pending = task_statuses == TaskStatus.PENDING
        task_is_created_before_current_time = task_times < self.time

        pending_indices = np.where(task_is_pending & task_is_created_before_current_time)[0]

        for index in pending_indices:
            task = self.tasks_list[index]
            if np.allclose(agent_pos, task.location, atol=0.1):
                task.service(self.time)
                tasks_serviced = True
        return tasks_serviced
    
    def _get_observation(self) -> dict:
        """
        Returns the current observation of the environment.

        Returns:
            dict: A dictionary containing the following information:
                - "position_agent": The position of the agent.
                - "position_task": The positions of the tasks.
                - "time_clock": The clocks in the environment.
                - "z": The progress made in the environment.
        """
        return {
            "position_agent" : self.agent_pos,
            "position_task" : self.tasks_pos,
            "time_clock" : self.clocks,
            "z" : self.progress
        }
    
    def _get_info(self) -> dict:
            """
            Returns a dictionary containing information about the current state of the environment.

            Returns:
                dict: A dictionary with the following keys:
                    - "Current Time in Seconds": The current time in seconds.
                    - "Number of Tasks Serviced": The number of tasks that have been serviced.
            """
            return {
                "Current Time in Seconds" : self.time,
                "Number of Tasks Serviced" : self.tasks_done,
            }
    
    def _move_agent(self, action: np.ndarray) -> np.ndarray:
        """
        Moves the agent based on the given action. **Used in the `step` method.**

        Args:
            action (numpy.ndarray): The action to be performed by the agent.

        Returns:
            numpy.ndarray: The new location of the agent after performing the action.

        Raises:
            ValueError: If the angular velocity is zero.
        """
        # Calculate the action (angular velocity) from the normalized action space
        scaled_action = action * self.MAX_ANGULAR_VELOCITY
        
        agent_location = np.copy(self.agent_pos)

        # Get the angular velocity from the action
        self.yaw_dot = scaled_action[0]

        if self.yaw_dot == 0:
            raise ValueError("Angular velocity must not be zero")

        # Calculate the change in x and y coordinates with velocity and angular velocity using the kinematic equations
        new_yaw = self.yaw + self.yaw_dot*self.dt
        self.dx = self.v * (np.sin(new_yaw) - np.sin(self.yaw)) / self.yaw_dot
        self.dy = self.v * (np.cos(self.yaw) - np.cos(new_yaw)) / self.yaw_dot
        agent_location[0] += self.dx
        agent_location[1] += self.dy

        # Update the yaw
        self.yaw = new_yaw

        # Check if the yaw is within the range [-pi, pi]
        if self.yaw > np.pi:
            self.yaw -= 2*np.pi
        if self.yaw < -np.pi:
            self.yaw += 2*np.pi
            
        # return agent_location
        return np.clip(agent_location, 0, self.size)
    
    def _update_clocks(self) -> np.ndarray:
        """
        Update the clocks of pending tasks. **Used in the `step` method.**

        This method updates the clocks of pending tasks based on the current time.
        It checks if a task is pending and if its creation time is less than or equal to the current time.
        If both conditions are met, the clock for that task is incremented by the time step.

        Returns:
            numpy.ndarray: A copy of the original clocks array with the updated values.
        """
        clocks = np.copy(self.clocks)
        if not self.tasks_list:
            return clocks

        # Find the indices of pending tasks with creation times less than to the current time; do not touch the task that is created at the current time
        pending_indices = [i for i, task in enumerate(self.tasks_list) if task.status == TaskStatus.PENDING and task.time_created < self.time]
        # Increment the clocks of the pending tasks
        clocks[pending_indices] += self.dt

        # Round the clocks to 1 decimal place
        clocks = np.round(clocks, 1)

        return clocks
    
    def _remove_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove serviced tasks from the task list and update the clocks. **Used in the `step` method.**

        Returns:
            Tuple[np.ndarray, np.ndarray]: The updated tasks positions and clocks arrays.
        """
        task_positions, clocks = np.copy(self.tasks_pos), np.copy(self.clocks)
        
        # Get indices of serviced tasks
        serviced_indices = np.where(np.fromiter((task.is_serviced for task in self.tasks_list), dtype=bool))[0]
        # Mark serviced tasks as removed in the task_positions array
        task_positions[serviced_indices * 2] = 0
        task_positions[serviced_indices * 2 + 1] = 0
        # Reset clocks associated with serviced tasks to 0
        clocks[serviced_indices] = 0
        return task_positions, clocks
    
    def _is_terminated(self) -> bool:
            """
            Check if the episode is terminated.

            The episode is considered terminated when all the tasks in the task list
            have been serviced.

            Returns:
                bool: True if all tasks have been serviced, False otherwise.
            """
            return all(task.is_serviced for task in self.tasks_list)
    
    def _is_truncated(self) -> bool:
        """
        Check if the episode is truncated.

        Returns:
            bool: True if the agent is truncated, False otherwise.
        """
        # create a bool variable to check if agent is out of bounds
        out_of_bounds = any(self.agent_pos < 0) or any(self.agent_pos > self.size)
        return self.progress[0] >= 1 or out_of_bounds
    
    def _get_reward(self, term: bool, trunc: bool) -> float:
        """
        Calculate the reward based on the current state of the environment.

        Parameters:
        - term (bool): Indicates whether the episode has terminated.
        - trunc (bool): Indicates whether the episode was truncated.

        Returns:
        - reward (float): The calculated reward value.
        """
        reward = 0
        if self.agent_on_task:
            reward += self._calculate_reward(max_wait_time=100)
        
        if term:
            reward += 1
        
        if trunc:
            reward = -1

        return float(reward)
       
    def _calculate_reward(self, max_wait_time: Union[int, float]) -> float:
        """
        Calculates the reward when agent has serviced tasks.

        Parameters:
        - max_wait_time (Union[int, float]): The maximum wait time for a task.

        Returns:
        - reward (float): The calculated reward.
        """
        reward = 0
        if not self.tasks_list:
            return reward

        # Calculate wait times for all serviced tasks at this step
        wait_times = np.array([task.wait_time() for task in self.tasks_list if task.status == TaskStatus.SERVICED and task.time_serviced == self.time])

        if len(wait_times) > 0:
            # Normalize wait times to [0, 1] range
            norm_wait_times = wait_times / max_wait_time

            # Calculate reward contributions using an exponential decay function
            reward_contributions = np.exp(-norm_wait_times)

            # Sum up reward contributions for all serviced tasks
            reward = np.sum(reward_contributions)
        
        return float(reward)
    
    def _render_frame(self):
        """
        Render the current frame of the environment.
        This method renders a frame of the environment using Seaborn.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        try:
            canvas = FigureCanvas(fig)
            ax.set_xlim(0, self.size)
            ax.set_ylim(0, self.size)
            ax.set_aspect('equal')

            # Draw the agent
            agent_x, agent_y = self.agent_pos[0], self.agent_pos[1]
            ax.plot(agent_x, agent_y, 'ro', label='Agent')
            t_idx = 0
            # Draw the tasks
            for task in self.tasks_list:
                if task.status == TaskStatus.PENDING and task.time_created <= self.time:
                    task_x, task_y = task.location[0], task.location[1]
                    color = self.task_colors.get(t_idx, np.random.rand(3,))
                    ax.plot(task_x, task_y, marker='o', color=color, label=f'Task {t_idx}')
                    self.task_colors[t_idx] = color
                t_idx += 1

            ax.legend(loc='best')
            ax.set_xticks(np.arange(0, self.size+1, 1))
            ax.set_yticks(np.arange(0, self.size+1, 1))
            ax.set_title(f"Time: {self.time:.1f}s")
            if self.render_mode == "human":
                # save the image as file name step_{i}.png where i is the step number
                # save inside the folder render_frames
                plt.savefig(f"render_frames/step_{self.steps}.png")
                del fig, ax, canvas
                gc.collect()
            elif self.render_mode == "rgb_array":
                # return the image as a numpy array
                canvas.draw()
                # get the np.array representation of the canvas
                frame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape(canvas.get_width_height()[::-1] + (3,))
                del fig, ax, canvas
                gc.collect()
                return frame
        finally:
            plt.close('all')

if __name__ == "__main__":
    import json
    env_params = "../env_config.json"
    with open(env_params, "r") as f:
        env_params = json.load(f)

    env = DVRPEnv(render_mode=None, **env_params)
    check_env(env, warn=True)
    env = FlattenObservation(env)

    obs, _ = env.reset()

    for t in env.unwrapped.tasks_list:  # ignore
        print(t)


    step = 0
    rewards = []
    # frames = []
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        # frame = env.render()
        # frames.append(frame)
        # del frame
        done = terminated or truncated
        step += 1
        if step == 50:
            pass
        # print all relevant information
        print(f"Step: {step}, Action: {action}, Term: {terminated}, Trunc: {truncated}, Done: {done}, Info: {info}")
        # print(env._is_agent_on_task())
        print(f"Current observation: {obs}")
        print(f"Total reward: {sum(rewards)}")
        print(f"-"*50)