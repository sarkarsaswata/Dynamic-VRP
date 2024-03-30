from enum import Enum
import numpy as np

class TaskStatus(Enum):
    PENDING = 0
    SERVICED = 1
    INCOMPLETE = 2

class Task:
    """Represents a task that needs to be serviced.

    Attributes:
        id (int): The unique identifier of the task.
        location (np.ndarray): The (x, y) coordinates of the task's location.
        time_created (float): The time at which the task was created.
        initial_wait (float): The initial wait time before the task can be serviced.
        service_time (float): The time it takes to service the task.
        status (TaskStatus): The current status of the task.
        time_serviced (float): The time at which the task was serviced.

    """

    def __init__(self, id:int, location:np.ndarray, time_created:float, initial_wait:float, service_time:float):
        self.id = id
        self.location = location    # (x, y) coordinates
        self.time_created = time_created
        self.initial_wait = initial_wait
        self.service_time = service_time
        self.status = TaskStatus.PENDING
        self.time_serviced = -1

    @property
    def is_pending(self) -> bool:
        """Check if the task is pending.

        Returns:
            bool: True if the task is pending, False otherwise.

        """
        return self.status == TaskStatus.PENDING
    
    @property
    def is_serviced(self) -> bool:
        """Check if the task has been serviced.

        Returns:
            bool: True if the task has been serviced, False otherwise.

        """
        return self.status == TaskStatus.SERVICED
    
    def service(self, time:float) -> None:
        """Mark the task as serviced at the given time.

        Args:
            time (float): The time at which the task is serviced.

        """
        self.status = TaskStatus.SERVICED
        self.time_serviced = time
    
    def incomplete(self, time:float) -> None:
        """Mark the task as incomplete.

        """
        self.time_serviced = time
        self.status = TaskStatus.INCOMPLETE
        
    def __str__(self) -> str:
        """Return a string representation of the task.

        Returns:
            str: A string representation of the task.

        """
        if self.status == TaskStatus.SERVICED:
            return f"Task {self.id} at {self.location} was created at time {self.time_created}, serviced at time {self.time_serviced}, and state is {self.status}"
        else:
            return f"Task {self.id} at {self.location} was created at time {self.time_created} and state is {self.status}"
    
    def wait_time(self) -> float:
        """Calculate the wait time of the task.

        Returns:
            float: The wait time of the task.

        Raises:
            ValueError: If the task has not been serviced yet.

        """
        if self.time_serviced == -1:
            raise ValueError("Task has not been serviced yet")
        return self.time_serviced - self.time_created