import gymnasium as gym
import numpy as np
import pygame
import gymnasium.spaces as spaces
from stable_baselines3.common.env_checker import check_env

class DVRPEnv(gym.Env):
    metadata = {'render_modes': ["human", "rgb_array"],
                'render_fps': 4
                }
    def __init__(self, env_size=10, task_arrival_rate=None, new_task_period=10,
                 render_mode=None, max_step=2, buffer_time=10):
        super(DVRPEnv, self).__init__()

        assert env_size > 0, "Environment size must be greater than 0"
        assert task_arrival_rate is not None and task_arrival_rate in [0.5, 0.6, 0.7, 0.8, 0.9], "Task arrival rate must be among [0.5, 0.6, 0.7, 0.8, 0.9]"
        assert new_task_period is not None and new_task_period > 0, f"New task period must be greater than 0, given is {new_task_period}"
        assert max_step is not None and max_step > 0, "Max step must be greater than 0"
        assert buffer_time is not None and buffer_time > 0, "Buffer time must be greater than 0"

        self.size = env_size
        self.window_size = 512          # The size of pygame window
        self.rate = task_arrival_rate

        self.max_tot_tasks = self._get_max_tasks()

        self.observation_space = spaces.Dict(
            {
                "agent_pos" : spaces.Box(low=0, high=self.size, shape=(2,), dtype=int),
                "tasks_pos" : spaces.Box(low=0, high=self.size - 1, shape=(self.max_tot_tasks*2,), dtype=int),
                "clocks" : spaces.Box(low=0, high=1000, shape=(self.max_tot_tasks,), dtype=float),
                "progress" : spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
            }
        )
        spaces.Dict.is_np_flattenable
        self.action_space = spaces.Discrete(8)
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.agent_on_task = False

        # period of new task arrival
        self.new_task_period = new_task_period
        # maximum steps in episode
        self.max_step = max_step
        # buffer time to complete the tasks
        self.buffer_time = buffer_time


    def reset(self, seed=None, options=None):
        self.np_random, SEED = gym.utils.seeding.np_random(seed)
        super().reset(seed=SEED, options=options)
        # time step in the episode
        self.time_step = 0
        # current task step starting from 0
        self.task_step = 0
        # total tasks serviced
        self.total_tasks_serviced = 0
        
        # UAV start position uniformly distributed in the grid
        self.agent = self.np_random.integers(0, self.size, size=(2, ), dtype=int)
        # depot position at the starting position of UAV, make a copy
        # self.depot = np.copy(self.agent)
        
        self.tasks = np.zeros((self.max_tot_tasks*2), dtype=int)
        self.clocks = np.zeros(self.max_tot_tasks, dtype=int)
        # generate the task locations randomly with a helper function
        self.tasks = self._generate_task_locs()
        # get the progress of the episode with a helper function
        self.progress = self._get_progress()
        # Check if the agent is on a task location
        self.agent_on_task = self._is_agent_on_task()
        observation = self._get_observation()
        info = self._get_info()

        # remder the environment
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        # store the action for rendering
        self.act = action
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f"Action {action} is invalid.")
        self.time_step += 1
        # move the UAV
        self.agent = self._move_agent(action)
        # update the task waiting times
        self.clocks = self._update_waiting_times()

        # check if agent reaches any goal location
        self.agent_on_task = self._is_agent_on_task()
        
        # if time step is a multiple of new task period, generate new tasks
        if self.time_step % self.new_task_period == 0:
            self.task_step += 1
            self.tasks = self._generate_task_locs()
        
        # get the progress of the episode with a helper function
        self.progress = self._get_progress()
        # check if terminated with a helper function
        terminated = self._is_terminated()
        # check if truncated with a helper function
        truncated = self._is_truncated()
        # get reward with a helper function
        reward = self._get_reward()

        # if agent is on task, remove the task
        if self.agent_on_task:
            self.total_tasks_serviced += 1
            self.tasks, self.clocks = self._remove_completed_tasks()
        
        # get observation with a helper function
        observation = self._get_observation()
        # get info with a helper function
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _get_max_tasks(self):
        return int(np.ceil(35 * self.rate))
    
    def _get_observation(self):
        return {
            "agent_pos" : self.agent,
            "tasks_pos" : self.tasks,
            "clocks" : self.clocks,
            "progress" : self.progress
        }
    
    def _get_info(self):
        return {
            "time step" : self.time_step,
            "is agent on task" : self.agent_on_task,
            "agent position" : self.agent,
            "task position" : self.tasks,
            "task clocks" : self.clocks,
            "progress" : self.progress
        }
    
    def _get_progress(self):
        low, high = -1.0, 1.0
        total_steps = self.max_step * self.new_task_period + self.buffer_time

        # Calculate the ratio of the current step to the total steps
        # make sure it is a float value
        progress = float(self.time_step / total_steps)
        progress = low + progress * (high - low)

        # print error if progress is not within the range (-1, 1)
        if not low <= progress <= high:
            raise ValueError(f"Progress {progress} is not within the range (-1, 1)")
        
        # Clip the progress to ensure it stays within the range (-1, 1)
        progress = np.clip(a=progress, a_min=low, a_max=high)

        return np.array([progress], dtype=float)
    
    def _is_agent_on_task(self):
        tasks = np.copy(self.tasks[self.tasks > 0]).reshape(-1, 2)
        return any(np.array_equal(self.agent, task_poses) for task_poses in tasks)
    
    def  _move_agent(self, action):
        move = self._action_to_move(action)
        low = np.array([0, 0])
        high = np.array([self.size, self.size])
        return self.agent + move
        # return np.clip(self.agent + move, low, high)
    
    def _action_to_move(self, action):
        # Define the mapping from actions to movements
        movements = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

        # Get the movement corresponding to the action
        move = movements[action]

        # Convert the movement to a numpy array and return it
        return np.array(move, dtype=int)
    
    def _update_waiting_times(self):
        tasks = np.copy(self.tasks[self.tasks > 0]).reshape(-1, 2)
        clocks = np.copy(self.clocks)
        clocks[:len(tasks)] += 1
        return clocks
    
    def _remove_completed_tasks(self):
        non_zero_tasks = np.trim_zeros(self.tasks)
        non_zero_clocks = np.trim_zeros(self.clocks)
        indices_to_remove = []
        for i, task in enumerate(non_zero_tasks.reshape(-1, 2)):
            if np.array_equal(self.agent, task):
                indices_to_remove.extend([i*2, i*2 + 1])
        
        non_zero_tasks = np.delete(non_zero_tasks, indices_to_remove)
        non_zero_clocks = np.delete(non_zero_clocks, [i//2 for i in indices_to_remove])
        tasks = np.zeros_like(self.tasks)
        clocks = np.zeros_like(self.clocks)
        tasks[:len(non_zero_tasks)] = non_zero_tasks
        clocks[:len(non_zero_clocks)] = non_zero_clocks
        return tasks, clocks
    
    def _is_terminated(self):
        total_time = self.max_step * self.new_task_period
        # if time step is greater than or equal to total time steps, then return True if no tasks present in tasks, else False
        return self.time_step >= total_time and np.all(self.tasks == 0)
    
    def _is_truncated(self):
        # Check if the agent is out of the environment
        out_of_bounds = any([
            self.agent[0] < 0,
            self.agent[1] < 0,
            self.agent[0] > self.size,
            self.agent[1] > self.size
        ])
        # if out_of_bounds:
        #     print(f"Agent is out of bounds at time step {self.time_step}: Agent location {self.agent}")

        # Return True if time step is greater than or equal to total time steps, or if the agent is out of bounds
        total_time = self.max_step * self.new_task_period + self.buffer_time
        return self.time_step >= total_time or out_of_bounds
    
    def _get_reward(self):
        # if action leads the agent location to a task location, return max of the clock
        # else return negative of the max of the clock
        reward = 0
        if not self.agent_on_task:
            reward = float(-np.max(self.clocks))
        else:
            if self._is_terminated():
                reward = 2000
            else:
                reward = float(np.max(self.clocks))
                # reward = reward

        # If episode is truncated, subtract 2000 from the reward
        if self._is_truncated():
            reward = reward - 2000

        return float(reward)
    
    def _get_poisson_value(self):
        return int(self.np_random.poisson(self.rate))
    
    def _get_task_locations(self, n_tasks):
        if n_tasks == 0:
            return np.array([])

        tasks = np.copy(self.tasks[self.tasks > 0])
        new_task_locations = []

        # Generate new task locations and append in the list
        for _ in range(n_tasks):
            new_task = np.copy(self.agent)
            while (
                np.array_equal(new_task, self.agent) or
                any(np.array_equal(new_task, task_poses) for task_poses in tasks.reshape(-1, 2))
            ):
                new_task = self.np_random.integers(low=1, high=self.size - 1, size=(2, ), dtype=int)
            new_task_locations.append(new_task)

        # convert the new task location list to numpy array 1D
        new_task_locations = np.array(new_task_locations).flatten()

        return new_task_locations
    

    def _generate_task_locs(self):
        tasks = np.copy(self.tasks)
        # if _generate_task_locs is called in the reset method, then get at least one task
        if self.task_step == 0:
            task_arrivals = self._get_poisson_value() + 1

            # if task_arrivals:
            #     print(f"Task arrivals at time step {self.time_step}: {task_arrivals}")
            
            new_task_locations = self._get_task_locations(n_tasks=task_arrivals)
            # update the task locations: first the incomplete tasks, then the new tasks
            current_task = np.append(tasks[tasks > 0], new_task_locations)
            tasks[:len(current_task)] = current_task
            return tasks
        
        elif self.task_step <= self.max_step:
            task_arrivals = self._get_poisson_value() + 1

            # if task_arrivals:
            #     print(f"Task arrivals at time step {self.time_step}: {task_arrivals}")
            
            new_task_locations = self._get_task_locations(n_tasks=task_arrivals)
            # update the task locations: first the incomplete tasks, then the new tasks
            current_task = np.append(tasks[tasks > 0], new_task_locations)
            tasks[:len(current_task)] = current_task
            return tasks
        
        else:
            return self.tasks
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size,
                                 self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )   # The size of a grid cell in pixels

        # First we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw the agent's last position
        # if self.time_step > 0:
        #     last_pos = self.agent - self._action_to_move(self.act)
        #     alpha = int(255 * 0.1) # 50% transparent, you can adjust the transparency here
        #     pygame.draw.circle(
        #         canvas,
        #         (0, 0, alpha),
        #         (last_pos + 0.5) * pix_square_size,
        #         pix_square_size / 3,
        #     )

        # Then we draw all the tasks locations
        tasks = np.copy(self.tasks[self.tasks != 0])
        for task in tasks.reshape(-1, 2):
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * task,
                    (pix_square_size, pix_square_size),
                ),
            )
        
        # Finally we add some grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1
            )
        
        # Add code here to render the time step
        pygame.font.init()
        font = pygame.font.Font(None, 36)
        text = font.render(f"Time step: {self.time_step}", True, (0, 0, 0))
        canvas.blit(text, (10, 10))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        else: # rgb array
            return np.transpose(
                np.array(
                    pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

# check the environment
if __name__ == "__main__":
    import json
    config_file = "/home/moonlab/MS_WORK/ms_thesis/dictpolicy/configs/config1.json"

    with open(config_file, "r") as file:
        params = json.load(file)

    # Create the environment with the parameters
    env = DVRPEnv(**params)
    obs, info = env.reset()

    done = False
    step = 0
    print(f"Initail observation: {obs} at time step {step}")
    print("-"*50)
    total_reward = 0
    env.metadata['render_fps'] = 1

    # list to store the observations as frames
    frames = []
    # list to store the rewards
    rewards = []
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, trucated, info = env.step(action)
        frme = env.render()
        frames.append(frme)
        rewards.append(reward)
        step += 1
        done = terminated or trucated
        print(f"Step: {step}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Observation: {obs}")
        print("-"*50)
    env.close()
    print(f"Total reward: {sum(rewards)}")
    from vid import create_video
    create_video(frames, "output_video.mp4", fps=2)
    # check the environment
    # check_env(env, warn=True)