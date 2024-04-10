"""
This file, env.py, implements the environment for the DVRP problem using the OpenAI Gym API.
"""

from typing import Tuple, Union
import gymnasium as gym
import gymnasium.spaces as spaces
import gymnasium.utils.seeding as seeding
import numpy as np
import pygame
from dvrp.utils.gen_tasks import PoissonTaskGenerator
from dvrp.utils.task import TaskStatus



class DVRPEnv(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array"],
                "render_fps": 4
    }
    def __init__(self, size=10, lam=0.5, n_tasks=10, max_time=200, render_mode=None):
        super(DVRPEnv, self).__init__()
        assert 0.5 <= lam and lam <= 0.9, "task arrival rate must be between 0.5 and 0.9"
        self.size = size
        self.lam = lam
        self.n_tasks = n_tasks
        self.max_time = max_time
        self.total_time = self.max_time * 2.5
        # action space: Discrete
        self.action_space = spaces.Discrete(8)
        # observation space: Dict
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=self.size, shape=(2,), dtype=np.float16),
            "tasks_pos": spaces.Box(low=0, high=self.size, shape=(self.n_tasks*2, ), dtype=np.float16),
            "clocks": spaces.Box(low=0, high=self.max_time*2, shape=(self.n_tasks,), dtype=np.float16),
            # "distances": spaces.Box(low=0, high=self.size*np.sqrt(2), shape=(self.n_tasks,), dtype=np.float16),
            "progress": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float16)
        })

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.window_size = 512          # The size of pygame window
        self.window = None
        self.pyclock = None
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        _, SEED = seeding.np_random(seed)
        super().reset(seed=SEED, options=options)
        self.steps = 0
        self.tasks_done = 0
        # Agent position is initialized at the center of the environment
        self.pos_agent = np.array([self.size/2, self.size/2], dtype=np.float16)
        # Generate tasks
        self.tasks_list = PoissonTaskGenerator(total_tasks=self.n_tasks, max_time=self.max_time, seed=SEED).get_tasks(self.lam)
        self.arrival_times = np.array([task.time_created for task in self.tasks_list], dtype=np.float16)
        # Initialize the tasks and clocks
        self.pos_tasks = self._get_tasks(self.steps)
        self.clocks = np.zeros((self.n_tasks), dtype=np.float16)
        self.distances = self._get_task_distances()
        self.progress = np.array([-1], dtype=np.float16)
        # get the initial observation
        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def reset_(self):
        self.tasks_list = self.tasks_list
        self.arrival_times = self.arrival_times
        self.steps = 0
        self.pos_agent = np.array([self.size/2, self.size/2], dtype=np.float16)
        self.pos_tasks = self._get_tasks(self.steps)
        self.clocks = np.zeros((self.n_tasks), dtype=np.float16)
        self.distances = self._get_task_distances()
        self.progress = np.array([-1], dtype=np.float16)
        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def _get_task_distances(self) -> np.ndarray:
        distances = np.zeros(self.n_tasks, dtype=np.float16)
        task_pos = self.pos_tasks.reshape((self.n_tasks, 2))
        non_zero_indices = np.any(task_pos != 0, axis=1)
        distances[non_zero_indices] = np.linalg.norm(self.pos_agent - task_pos[non_zero_indices], axis=1)
        return distances
    
    def step(self, action):
        self.steps += 1
        self.pos_agent = self._move_agent(action)
        self.clocks = self._update_clocks(time=self.steps)
        self.agent_on_task = self._check_agent_on_task()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        reward = self._get_reward3(terminated, truncated)

        if self.agent_on_task:
            self.pos_tasks, self.clocks = self._remove_task()
            self.tasks_done = sum(task.status == TaskStatus.SERVICED for task in self.tasks_list)
        
        self.pos_tasks = self._get_tasks(self.steps)
        self.distances = self._get_task_distances()
        self.progress = self._get_progress(self.steps)

        
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info
    
    def _get_tasks(self, steps):
        if steps == 0:
            task_pos = np.zeros((self.n_tasks*2), dtype=np.float16)
        elif hasattr(self, 'pos_tasks') and self.pos_tasks is not None:
            task_pos = np.copy(self.pos_tasks)
        matching_indices = np.where(self.arrival_times == steps)[0]

        if matching_indices.size == 0:
            return task_pos
        
        for index in matching_indices:
            task = self.tasks_list[index]
            x_index = index*2
            y_index = x_index + 1
            task_pos[x_index] = task.location[0]
            task_pos[y_index] = task.location[1]
        
        return task_pos
    
    def _get_obs(self):
        pos_tasks = np.concatenate((self.pos_tasks[self.pos_tasks != 0], self.pos_tasks[self.pos_tasks == 0]))
        clocks = np.concatenate((self.clocks[self.clocks != 0], self.clocks[self.clocks == 0]))
        return {
            "agent_pos": self.pos_agent,
            "tasks_pos": pos_tasks,
            "clocks": clocks,
            "progress": self.progress,
            # "distances": self.distances
        }
    
    def _get_info(self):
        return {
            "tasks_done": self.tasks_done
        }
    
    def _move_agent(self, action):
        move = self._action_to_move(action)
        agent_pos = self.pos_agent + move
        # agent_pos = np.clip(agent_pos, 0, self.size-1)
        return agent_pos
    
    def _action_to_move(self, action):
        # Define the mapping from actions to movements
        movements = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        # Get the movement corresponding to the action
        move = movements[action]
        return np.array(move, dtype=np.float16)
    
    def _check_agent_on_task(self) -> bool:
        task_serviced = False

        for task in self.tasks_list:
            if task.is_pending and (task.location == self.pos_agent).all() and task.time_created < self.steps:
                task.service_task(self.steps)
                task_serviced = True
                
        return task_serviced
    
    def _remove_task(self):
        task_pos, clocks = np.copy(self.pos_tasks), np.copy(self.clocks)
        # get indices of serviced tasks
        serviced_indices = np.where(np.fromiter((task.is_serviced for task in self.tasks_list), dtype=bool))[0]
        # mark serviced tasks as removed in the tasks pos array
        task_pos[serviced_indices*2] = 0
        task_pos[serviced_indices*2+1] = 0
        # Reset the clocks of serviced tasks to 0
        clocks[serviced_indices] = 0
        return task_pos, clocks
    
    def _update_clocks(self, time) -> np.ndarray:
        clocks = np.copy(self.clocks)
        # Find the indices of pending tasks with creation times less than the current step; not touching the task that is created at the current step
        pending_indices = [i for i, task in enumerate(self.tasks_list) if task.is_pending and task.time_created < time]
        # Increment the clocks of the pending tasks
        clocks[pending_indices] += 1
        return clocks
    
    # def _get_progress(self, steps) -> np.ndarray:
    #     min, max = -1, 1
    #     progress = float(steps/self.total_time)
    #     progress = min + (progress * (max - min))
    #     progress = round(progress, 4)
    #     return np.array([progress], dtype=np.float16)
    
    def _get_progress(self, steps) -> np.ndarray:
        return np.array([steps], dtype=np.float16)
    
    def _is_terminated(self) -> bool:
        return self.tasks_done == self.n_tasks
    
    def _is_truncated(self) -> bool:
        # check if the agent is out of the boundary of arena
        out_of_bounds = np.any(self.pos_agent < 0) or np.any(self.pos_agent >= self.size)
        return self.steps >= self.total_time or out_of_bounds
        # return self.steps >= self.total_time
    
    def _get_reward1(self, terminated, truncated) -> float:
        reward = 0
        if not self.agent_on_task:
            reward -= np.max(self.clocks)
        else:
            if terminated:
                reward = 2000
            else:
                # reshape pos_tasks into a 2D array of shape (10, 2)
                pos_tasks_reshaped = self.pos_tasks.reshape(-1, 2)
                # find the task index the agent is currently servicing
                task_indices = np.argwhere((pos_tasks_reshaped == self.pos_agent).all(axis=1)).flatten()
                if task_indices.size > 0:  # check if there's a match
                    task_index = task_indices[0]
                    # get the clock value with the task index
                    reward = self.clocks[task_index]

        if truncated:
            reward -= 1000

        return float(reward)
    
    
    def _get_reward2(self, terminated, truncated) -> float:
        reward = 0
        if self.agent_on_task:
            reward += self._calculate_waittime_reward(max_wait_time=30)
            return float(reward)
        else:
            reward -= np.max(self.clocks)
        
        if terminated:
            # reward += self._calculate_completion_reward(time=self.steps)
            reward += 2000
            return float(reward)
        
        if truncated:
            reward = -2000
            return float(reward)
        
        return float(reward)
    
    def _get_reward3(self, terminated, truncated) -> float:
        reward = 0
        if self.agent_on_task:
            reward += self._calculate_waittime_reward(max_wait_time=50)
        else:
            # reward -= 0.001*np.max(self.clocks)
            reward = 0

        if terminated:
            if self.tasks_done > 0:
                reward = 20
        elif truncated:
            if self.tasks_done != 0:
                reward = -(self.n_tasks - self.tasks_done)/self.n_tasks
            else:
                reward = -1
        
        return float(reward)
    
    def _calculate_waittime_reward(self, max_wait_time) -> float:
        reward = 0
        wait_times = []
        # get the task.wait_time if the task is serviced at the current step
        for task in self.tasks_list:
            if (task.is_serviced and task.time_serviced == self.steps):
                wait_times.append(task.wait_time)
        wait_times = np.array(wait_times)  # move this line outside of the for loop
        if len(wait_times) > 0:
            normalized_wait_times = 1 - wait_times / max_wait_time
            reward = np.sum(np.exp(normalized_wait_times))*2
            return float(reward)
        else:
            return reward
    
    # def _calculate_completion_reward(self, time) -> float:
    #     # design a linear decay function for the reward as a function of termination time
    #     # max reward is 1, min reward is 0
    #     min, max = 0, 1
    #     reward = float(time/self.total_time)
    #     reward = 1 - (min + (max - min) * reward)
    #     return reward
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.pyclock is None:
            self.pyclock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size // self.size  # size of a grid cell in pixels

        # Draw the agent
        agent_pos = self.pos_agent.astype(int)
        if 0 <= agent_pos[0] < self.size and 0 <= agent_pos[1] < self.size:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (agent_pos + 0.5) * pix_square_size, # type: ignore
                pix_square_size / 3, # ensure agent fits within the grid cell
            )

        # Draw the tasks that are pending and time created is less than or equal to the current step
        for task in self.tasks_list:
            task_pos = task.location.astype(int)
            if task.is_pending and task.time_created <= self.steps:
                pygame.draw.rect(
                    canvas,
                    (255, 0, 0),
                    pygame.Rect(
                        pix_square_size * task_pos, # type: ignore
                        (pix_square_size, pix_square_size),
                    ),
                )

        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        # Add code here to render the time step
        pygame.font.init()
        font = pygame.font.Font(None, 36)
        text = font.render(f"Time step: {self.steps}", True, (0, 0, 0))
        canvas.blit(text, (10, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.pyclock.tick(self.metadata["render_fps"])
        else:   # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


if __name__ == "__main__":
    env = DVRPEnv(lam=0.7, render_mode=None)
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    # env = FlattenObservation(env)

    obs, _ = env.reset()

    for t in env.tasks_list:  # ignore
        print(t)

    step = 0
    rewards = []
    frames = []
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        frame = env.render()
        frames.append(frame)
        del frame
        done = terminated or truncated
        step += 1
        # print all relevant information
        print(f"Step: {step}, Action: {action}, Reward: {reward}, Term: {terminated}, Trunc: {truncated}, Done: {done}, Info: {info}")
        # print(env._is_agent_on_task())
        print(f"Current observation: {obs}")
        print(f"Total reward: {sum(rewards)}")
        print(f"-"*50)
    
    # from gymnasium.utils.save_video import save_video
    # save_video(frames=frames, video_folder='simulations', name_prefix='dvrp', fps=2)

    for t in env.tasks_list:
        print(t)
    print()
    print(f"Total tasks done: {env.tasks_done}")