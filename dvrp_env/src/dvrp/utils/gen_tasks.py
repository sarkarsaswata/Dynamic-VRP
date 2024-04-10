# General overview: This file contains the PoissonTaskGenerator class, which is used to generate tasks based on a model Poisson process with specified parameter of task arrival rate.
from typing import Tuple, Optional, List
import gymnasium as gym
import numpy as np
import gymnasium.utils.seeding as seeding
from dvrp.utils.task import Task

class PoissonTaskGenerator:
    """
    Generates tasks based on a Poisson process with specified parameters.

    Args:
        min (float, optional): Minimum value for task location. Defaults to 0.
        total_tasks (int, optional): Total number of tasks to generate. Defaults to 10.
        max_initial_wait (float, optional): Maximum initial wait time for tasks. Defaults to 0.
        service_time (float, optional): Average service time for tasks. Defaults to 0.
        env_size (float, optional): Size of the environment for task location. Defaults to 10.
        initial_tasks (int, optional): Number of initial tasks. Defaults to 1.
        max_time (float, optional): Maximum simulation time. Defaults to None.
        seed (int, optional): Seed for random number generation. Defaults to None.
    """

    def __init__(self, min: float = 0, total_tasks: int = 10, max_initial_wait: float = 0, service_time: float = 0, env_size: float = 10, initial_tasks: int = 1, max_time: Optional[float] = None, seed: Optional[int] = None) -> None:
        self.min: float = min
        self.total_tasks: int = total_tasks
        self.max_initial_wait: float = max_initial_wait
        self.service_time: float = service_time
        self.env_size: float = env_size
        self.initial_tasks: int = initial_tasks
        self.max_time: Optional[float] = max_time
        self.seed: Optional[int] = seed

        self.reset()
        
    def reset(self) -> None:
        """
        Reset the random number generator.
        """
        self.generator, self.seed = seeding.np_random(self.seed)

    def _get_location(self) -> np.ndarray:
        """
        Generate a random location for a task within the environment size.

        Returns:
            np.ndarray: Random (x, y) coordinates for task location.
        """
        # x = round(self.generator.uniform(self.min+1, self.env_size-1), 1)
        x = round(self.generator.uniform(self.min+1, self.env_size-1), 0)
        # y = round(self.generator.uniform(self.min+1, self.env_size-1), 1)
        y = round(self.generator.uniform(self.min+1, self.env_size-1), 0)
        return np.array([x, y], dtype=np.float16)
    
    def get_tasks(self, lam: float) -> List[Task]:
        """
        Generates tasks based on a Poisson process with the given arrival rate.

        Args:
            lam (float): Arrival rate parameter for the Poisson process.

        Returns:
            List[Task]: List of generated tasks.
        """
        first_time = round(self.generator.exponential(1/lam)*5, 0)
        tasks = []

        for _ in range(self.initial_tasks):
            new_task = Task(
                id = len(tasks),
                location = self._get_location(),
                time_created = 0,
                initial_wait=0,
                service_time = self.generator.normal(loc=self.service_time, scale=0.1*self.service_time)
            )
            tasks.append(new_task)
        
        sim_time = first_time
        while True:
            next_time = round(self.generator.exponential(1/lam)*5, 0)
            new_task = Task(
                id = len(tasks),
                location = self._get_location(),
                time_created = sim_time,
                initial_wait=0,
                service_time = self.generator.normal(loc=self.service_time, scale=0.1*self.service_time)
            )
            tasks.append(new_task)
            sim_time += next_time

            if self.max_time is not None and sim_time > self.max_time:
                    break
            if len(tasks) >= self.total_tasks:
                    break
        if len(tasks) < self.total_tasks:
             return self.get_tasks(lam=lam)
        return tasks

if __name__ == "__main__":
    generator = PoissonTaskGenerator(min=0, total_tasks=10, max_initial_wait=0, service_time=0, env_size=10, initial_tasks=1, max_time=200)
    tasks = generator.get_tasks(lam=0.5)
    for task in tasks:
         print(task)