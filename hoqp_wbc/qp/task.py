import dataclasses
from abc import abstractmethod

from hoqp_wbc.utils.common_types import *


@dataclasses.dataclass
class Task:
    """
    Define a task. Task is composed of equality and inequality constraints, the target is to minimize the slack
    variables.
    Task has priorities, smaller number means higher priority. Task with lower priority must have their
    solution being solved inside the null space of the tasks of higher task priorities.

    Ax - b =  w
    Dx - f <= v
    """
    A: Matrix
    b: Vector

    D: Matrix
    f: Vector

    @staticmethod
    def EMPTY(n_des: int):
        return Task(np.zeros((0, n_des)), np.zeros(0), np.zeros((0, n_des)), np.zeros(0))

    def update(self, task) -> None:
        self.A = task.A
        self.b = task.b
        self.D = task.D
        self.f = task.f

    def is_valid(self, n_des: int) -> bool:
        return (self.A is not None and self.b is not None) \
            or (self.D is not None and self.f is not None) \
            and (self.A.shape[1] == n_des and self.D.shape[1] == n_des) \
            and (self.A.shape[0] == self.b.shape[0] and self.D.shape[0] == self.f.shape[0])

    def __add__(self, other):
        if other.A is None and other.b is None and other.D is None and other.f is None:
            return Task(**dataclasses.asdict(self))
        return Task(
            np.concatenate([self.A, other.A]),
            np.concatenate([self.b, other.b]),
            np.concatenate([self.D, other.D]),
            np.concatenate([self.f, other.f]),
        )
