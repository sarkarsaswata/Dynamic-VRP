"""
This file, env.py, implements the environment for the DVRP problem using the OpenAI Gym API.
"""

from typing import Tuple, Union
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pygame
from gymnasium.utils import seeding
from stable_baselines3.common.env_checker import check_env
from task import TaskStatus
from gen_tasks import PoissonTaskGenerator

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
        self.max_time = global_max_step
        self.window_size = 512  # The size of the pygame window

        # action space: angular velocity of the agent
        action_low = np.array([-20*np.pi/180])
        action_high = np.array([20*np.pi/180])
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(1, ), dtype=np.float32)

        # observation space
        self.observation_space = spaces.Dict(
            {
                "agent_pos" : spaces.Box(low=0, high=self.size, shape=(2, ), dtype=np.float32),
                "tasks_pos" : spaces.Box(low=0, high=self.size, shape=(self.max_tasks*2, ), dtype=np.float32),
                "clocks" : spaces.Box(low=0, high=self.max_time, shape=(self.max_tasks, ), dtype=np.float32),
                "progress" : spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32)
            }
        )

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.window = None
        self.pygame_clock = None

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int): Seed for the random number generator. If None, a random seed will be used.
            options (dict): Additional options for resetting the environment.

        Returns:
            obs (object): The initial observation of the environment.
            info (dict): Additional information about the environment's state.
        """
        self.np_random, SEED = seeding.np_random(seed)
        super().reset(seed=SEED, options=options)
        self.time = 0           # Current time in seconds
        self.steps = 0          # number of steps taken by the agent
        self.v = 1              # velocity of the agent to 1
        self.yaw = 0            # angular velocity of the agent to 0
        self.dt = 0.1           # delta time 0..1s for executing the action
        self.tasks_done = 0     # number of tasks serviced by the agent

        # Agent's initial position is the middle of the environment
        self.agent_pos = np.array([self.size // 2, self.size // 2], dtype=np.float32)

        # generate tasks using the poisson task generator and store the arrival times
        self.tasks_list = PoissonTaskGenerator(total_tasks=self.max_tasks, max_time=self.max_time, seed=SEED).get_tasks(lam=self.rate)
        self.arrival_times = np.array([task.time_created for task in self.tasks_list], dtype=np.float32).reshape(-1)

        # Initialize the tasks, clocks, and progress arrays
        self.tasks_pos = self._get_tasks(time=self.time)
        self.clocks = self._get_clocks()
        self.progress = self._get_progress(time=self.time)
        
        self.agent_on_task = self._is_agent_on_task()

        obs = self._get_obs()
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
        # increase the time step with self.dt
        self.time += self.dt        # Current time in seconds, use it to fetch the tasks and get progress
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
        reward = self._get_reward(term=terminated, trunc=truncated)
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def _get_tasks(self, time:float) -> np.ndarray:
        """
        Get the locations of tasks at a given time. Initiated in the reset method.

        Parameters:
        - time (float): Current time.

        Returns:
        - tasks_locs (numpy.ndarray): An array containing the locations of tasks at the given time.
        """
        try:
            tasks_locs = np.copy(self.tasks_pos)
        except AttributeError:
            tasks_locs = np.zeros((self.max_tasks*2, ), dtype=np.float32)

        if time not in self.arrival_times:
            return tasks_locs
        else:
            matching_indices = np.where(self.arrival_times == time)[0]
            for index in matching_indices:
                task = self.tasks_list[index]
                tasks_locs[index*2] = task.location[0]
                tasks_locs[index*2+1] = task.location[1]
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
        AssertionError: If the calculated progress is not within the range [-1, 1].
        """
        low, high = -1.0, 1.0
        prog = float(time / self.max_time)
        prog = low + prog * (high - low)
        assert -1 <= prog <= 1, f"Progress must be in the range [-1, 1], but got {prog}"
        return np.array([prog], dtype=np.float32)
    
    def _is_agent_on_task(self) -> bool:
        """
        Checks if the agent is on a task location and services the pending tasks. Initiated in the reset method.

        Returns:
            bool: True if there are tasks serviced, False otherwise.
        """
        tasks_serviced = False
        agent_pos = self.agent_pos
        current_time = self.time

        task_locations = np.array([task.location for task in self.tasks_list], dtype=np.float32)
        task_statuses = np.array([task.status for task in self.tasks_list])

        pending_indices = np.where((task_statuses == TaskStatus.PENDING) &
                                      (task_locations[:, 0] == agent_pos[0]) &
                                        (task_locations[:, 1] == agent_pos[1]) &
                                        (np.array([task.time_created for task in self.tasks_list]) < current_time))[0]
        
        for index in pending_indices:
            self.tasks_list[index].service(current_time)
            tasks_serviced = True
        return tasks_serviced
    
    def _get_obs(self) -> dict:
        """
        Returns the current observation of the environment.

        Returns:
            dict: A dictionary containing the following information:
                - "agent_pos": The position of the agent.
                - "tasks_pos": The positions of the tasks.
                - "clocks": The clocks in the environment.
                - "progress": The progress made in the environment.
        """
        return {
            "agent_pos" : self.agent_pos,
            "tasks_pos" : self.tasks_pos,
            "clocks" : self.clocks,
            "progress" : self.progress
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
        """
        #CHECKED
        agent_location = np.copy(self.agent_pos)
        self.yaw_dot = action[0]
        self.dx = (self.v*(np.sin(self.yaw + self.yaw_dot*self.dt) - np.sin(self.yaw)))/self.yaw_dot
        self.dy = (self.v*(np.cos(self.yaw) - np.cos(self.yaw + self.yaw_dot*self.dt)))/self.yaw_dot
        self.yaw += self.yaw_dot*self.dt
        agent_location[0] += self.dx
        agent_location[1] += self.dy
        return agent_location
    
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
        # Get the creation times of pending tasks
        pending_creation_times = np.array([task.time_created for task in self.tasks_list if task.status == TaskStatus.PENDING], dtype=np.float32)
        # Find the indices of pending tasks with creation times less than to the current time; do not touch the task that is created at the current time
        pending_indices = np.where(pending_creation_times < self.time)[0]
        # Increment the clocks of the pending tasks
        clocks[pending_indices] += self.dt

        return clocks
    
    def _remove_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove serviced tasks from the task list and update the clocks. **Used in the `step` method.**

        Returns:
            np.ndarray: The updated tasks and clocks arrays.
        """
        tasks, clocks = np.copy(self.tasks_pos), np.copy(self.clocks)
        # Get indices of serviced tasks
        serviced_indices = np.where([task.status == TaskStatus.SERVICED for task in self.tasks_list])[0]
        # Mark serviced tasks as removed in the tasks array
        tasks[serviced_indices * 2] = 0
        tasks[serviced_indices * 2 + 1] = 0
        # Reset clocks associated with serviced tasks to 0
        clocks[serviced_indices] = 0
        return tasks, clocks
    
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
        out_of_bounds = not (0 <= self.agent_pos[0] <= self.size and 0 <= self.agent_pos[1] <= self.size)
        return self.progress[0] == 1 or out_of_bounds
    
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
            reward = self._calculate_reward(max_wait_time=100)
        
        if term:
            reward += 1
        
        if trunc:
            reward -= 1

        return float(reward)
       
    def _calculate_reward(self, max_wait_time: Union[int, float]) -> float:
        """
        Calculates the reward when agent has serviced tasks.

        Returns:
            float: The calculated reward.
        """
        reward = 0
        # Get all the tasks that are serviced at this step
        serviced_tasks = [task for task in self.tasks_list if task.status == TaskStatus.SERVICED and task.time_serviced == self.time]
        if serviced_tasks:
            # Calculate wait time for all serviced tasks
            wait_times = np.array([task.wait_time() for task in serviced_tasks])

            # Normalize wait times to [0, 1] range
            norm_wait_times = (wait_times - 0) / max_wait_time

            # Calculate reward contributions using an exponential decay function
            reward_contributions = np.exp(-norm_wait_times)

            # Sum up reward contributions for all serviced tasks
            reward = np.sum(reward_contributions)
        
        return float(reward)
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.pygame_clock is None and self.render_mode == "human":
            self.pygame_clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pixel_size = self.window_size // self.size

        # Draw the agent
        agent_pos_int = tuple(map(int, self.agent_pos))  # Convert agent position to integers for rendering
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (int((agent_pos_int[0] + 0.5) * pixel_size), int((agent_pos_int[1] + 0.5) * pixel_size)),
            pixel_size // 2
        )

        tasks = np.copy(self.tasks_pos[self.tasks_pos != 0]).reshape(-1, 2)
        for t in tasks:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pixel_size * t[0],  # left
                    pixel_size * t[1],  # top
                    pixel_size,  # width
                    pixel_size  # height
                )
            )
        
        # Add code here to render the time
        pygame.font.init()
        font = pygame.font.SysFont("Roboto", 36)
        text = font.render(f"Time: {self.time:.2f}", True, (0, 0, 0))
        canvas.blit(text, (10, 10))

        if self.render_mode == "human":
            if self.window is not None:
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
                self.pygame_clock = pygame.time.Clock()
                self.pygame_clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(
                    pygame.surfarray.array3d(canvas)
                ), axes=(1, 0, 2)
            )

if __name__ == "__main__":
    import json
    env_params = "/home/moonlab/MS_WORK/ms_thesis/continuous/configs/env_config.json"
    with open(env_params, "r") as f:
        env_params = json.load(f)

    env = DVRPEnv(**env_params)
    obs, info = env.reset()
    done = False
    step = 0

    for t in env.tasks_list:
        print(t)
    
    print(f"initial Observation: {obs} at time {info['Current Time in Seconds']}")
    print("-"*50)

    total_reward = 0
    # Check the environment
    # check_env(env)
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        print(f"Step: {step}")
        print(f"Observation: {obs}")
        print(terminated, truncated)
        step += 1
        print("-"*50)
    