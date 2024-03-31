import time
from typing import List, Callable, Optional

import qpsolvers
import scipy

from qb_wbc.utils.common_types import *
from .task import Task


class HoQPLevel:
    def __init__(self, task: Task, higher_level=None) -> None:
        self.tasks = task
        self.higher_level = higher_level

        self.nv = None
        self.nz = None
        self.v_p_star = None
        self.x_star = None
        self.Z_p = None
        self.z_p: Vector = None
        self.v_p: Vector = None
        self.stacked_tasks_prev = None
        self.stacked_tasks = None

        self.H: Matrix = None
        self.c: Vector = None
        self.D: Matrix = None
        self.f: Vector = None

        self.Z_p_plus: Matrix = None

        self.previous_sol = None

    def _formulate(self) -> None:
        self.nv = self.tasks.D.shape[0]
        self.has_eq_constraint = self.tasks.A.shape[0] > 0
        self.has_ineq_constraint = self.nv > 0
        if self.higher_level:
            self.Z_p = self.higher_level.Z_p_plus
            self.v_p_star = self.higher_level.v_p
            self.x_star = self.higher_level.get_solution()
            self.nz = self.higher_level.Z_p_plus.shape[1]
            self.stacked_tasks_prev = self.higher_level.tasks
        else:
            self.nz = self.tasks.A.shape[1]
            self.Z_p = np.identity(self.nz)
            self.v_p_star = np.zeros(0)
            self.x_star = np.zeros(self.nz)
            self.stacked_tasks_prev = Task.EMPTY(
                self.tasks.A.shape[1] if self.tasks.A is not None else self.tasks.D.shape[1])
        self.stacked_tasks = self.tasks + self.stacked_tasks_prev

        self._build_h()
        self._build_c()
        self._build_d()
        self._build_f()

    def _solve(self) -> bool:
        self.previous_sol = qpsolvers.solve_qp(
            P=self.H, q=self.c, G=self.D, h=self.f,
            solver="osqp", polish=True, check_termination=25
        )

        if self.previous_sol is None:
            return False

        self.z_p = self.previous_sol[:self.nz]
        self.v_p = self.previous_sol[self.nz:self.nz + self.nv]

        return True

    def _post_process(self) -> None:
        if self.has_eq_constraint:
            kernel = scipy.linalg.null_space(self.tasks.A @ self.Z_p)
            self.Z_p_plus = self.Z_p @ kernel
        else:
            self.Z_p_plus = self.Z_p

        if self.has_ineq_constraint:
            self.v_p_star = np.concatenate([self.v_p_star, self.v_p])
        else:
            self.v_p_star = self.v_p

    def _build_h(self) -> None:
        if self.has_eq_constraint:
            t = self.tasks.A @ self.Z_p
            temp = t.T @ t
        else:
            temp = np.zeros((self.nz, self.nz))

        self.H = np.block([
            [temp, np.zeros((self.nz, self.nv))],
            [np.zeros((self.nv, self.nz)), np.eye(self.nv)]
        ])

    def _build_c(self) -> None:
        self.c = np.concatenate([
            self.Z_p.T @ self.tasks.A.T @ (self.tasks.A @ self.x_star - self.tasks.b),
            np.zeros(self.nv)
        ])

    def _build_d(self) -> None:
        # upside down, as concatenate add to bottom instead of top
        prev_D = self.stacked_tasks_prev.D
        curr_D = self.tasks.D if self.has_eq_constraint else np.zeros((0, self.nz))
        self.D = np.block([
            [np.zeros((self.nv, self.nz)), -np.eye(self.nv)],
            [prev_D @ self.Z_p, np.zeros((prev_D.shape[0], self.nv))],
            [curr_D @ self.Z_p, -np.eye(self.nv)],
        ])

    def _build_f(self) -> None:
        f_minus_dp_xstar = self.tasks.f - self.tasks.D @ self.x_star if self.has_ineq_constraint else np.zeros(0)
        prev_D = self.stacked_tasks_prev.D
        prev_f = self.stacked_tasks_prev.f
        self.f = np.concatenate([
            np.zeros(self.nv),
            prev_f - prev_D @ self.x_star + self.v_p_star,
            f_minus_dp_xstar
        ])

    def get_solution(self) -> Vector:
        return self.x_star + self.Z_p @ self.z_p

    def solve(self) -> bool:
        self._formulate()
        if self._solve():
            self._post_process()
            return True
        else:
            return False


class HoQP:
    """
    Hierarchical Optimization Quadratic Program.
    The goal is to solve a cascaded system of QP, the solution of each problem should lies in the null space of the
    previous problem, defined by A and D matrix.
    """

    def __init__(self, n_des: int):
        self.task_levels: List[List[Callable[[], Task]]] = []
        self.hoqp_levels: List[HoQPLevel] = []
        self.lowest_priority: int = -1

        self.n_des = n_des  # number of decision variables inside x, the target solution vector.
        self.n_tasks: int = 0
        self.solve_count = 0
        self.verbose = True

    def add_task(self, priority: int, task: Callable[[], Task]) -> None:
        if priority < 0 or priority > self.lowest_priority + 1:
            raise Exception("Priority must be sequential, and should be a natural number.")

        if priority > self.lowest_priority:
            self.lowest_priority = priority
            if priority == 0:
                self.hoqp_levels.append(HoQPLevel(Task.EMPTY(self.n_des), None))
            else:
                self.hoqp_levels.append(HoQPLevel(Task.EMPTY(self.n_des), self.hoqp_levels[priority - 1]))
            self.task_levels.append([])
        self.task_levels[priority].append(task)
        self.n_tasks += 1

    def add_tasks(self, priority: int, tasks: List[Callable[[], Task]]):
        for task in tasks:
            self.add_task(priority, task)

    def update_n_decision(self, n_des: int) -> None:
        self.n_des = n_des
        for level in self.hoqp_levels:
            level.tasks = Task.EMPTY(n_des)

    def solve(self) -> Optional[Vector]:
        if self.solve_count == 0:
            self.summary()

        start_time = time.time()
        failed = False
        for priority in range(self.lowest_priority + 1):
            collected_task = Task.EMPTY(self.n_des)
            for formulate_task in self.task_levels[priority]:
                collected_task += formulate_task()

            self.hoqp_levels[priority].tasks.update(collected_task)
            assert self.hoqp_levels[priority].tasks.is_valid(self.n_des)
            if not self.hoqp_levels[priority].solve():
                failed = True
                break

        sol = None if failed else self.hoqp_levels[-1].get_solution()
        if self.verbose:
            print(f"HoQP Problem {self.solve_count} solved {'success' if not failed else 'FAILED'}.")
            print(f"\t Time taken: {(time.time() - start_time) * 1000:.2f} ms.")
            if not failed:
                for level in self.task_levels:
                    for task in level:
                        t: Task = task()
                        eq_r, ueq_s = t.get_fitness(sol)
                        print(eq_r, ueq_s)

        self.solve_count += 1

        return sol

    def summary(self) -> None:
        print("Solving HoQP.")
        print(f"\tNumber of Priorities: {len(self.task_levels)}")
        print(f"\tNumber of Tasks: {self.n_tasks}")
        print(f"\tNumber of Decision Variables: {self.n_des}")
