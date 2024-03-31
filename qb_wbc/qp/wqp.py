from typing import List, Callable, Optional, Tuple

import osqp

from qb_wbc.utils.common_types import *
from .task import Task


class WQP:
    def __init__(self, n_des: int):
        self.weighted_tasks: List[Tuple[Callable[[], Task], float]] = []
        self.constraints: List[Callable[[], Task]] = []

        self.n_des = n_des  # number of decision variables inside x, the target solution vector.
        self.n_tasks: int = 0
        self.solve_count = 0
        self.verbose = True

        self.problem: Optional[osqp.OSQP] = None
        self.sol: Optional[Vector] = None

    def _formulate(self):
        constraints = Task.EMPTY(self.n_des)
        for ct in self.constraints:
            constraints += ct()
        n_constraints = constraints.b.shape[0] + constraints.f.shape[0]
        if n_constraints > 0:
            A = np.vstack([constraints.A, constraints.D])
            lb = np.concatenate([constraints.b, np.ones(constraints.f.shape[0]) * -np.infty])
            ub = np.concatenate([constraints.b, constraints.f])
        else:
            A = None
            lb = None
            ub = None

        weighted_task = Task.EMPTY(self.n_des)
        for wt in self.weighted_tasks:
            weighted_task += wt[0]() * wt[1]
        H = weighted_task.A.T @ weighted_task.A
        g = -weighted_task.A.T @ weighted_task.b

        if not self.problem:
            self.problem = osqp.OSQP()
            H = csc_matrix(H)
            A = csc_matrix(A)
            self.problem.setup(P=H, q=g, A=A, l=lb, u=ub)
        else:
            self.problem.update(q=g, l=lb, u=ub, Px=H, Ax=A)
        self.problem.update_settings(polish=True, polish_refine_iter=2, check_termination=20)

    def _solve(self) -> bool:
        self.sol = self.problem.solve()
        return np.all(self.sol)

    def add_weighted_task(self, wt: Callable[[], Task], weight: float) -> None:
        self.weighted_tasks.append((wt, weight))

    def add_constraint(self, ct: Callable[[], Task]) -> None:
        self.constraints.append(ct)

    def update_n_decision(self, n_des: int) -> None:
        self.n_des = n_des

    def get_solution(self) -> Optional[Vector]:
        return self.sol.x if self.sol is not None else None

    def solve(self) -> bool:
        self._formulate()
        return self._solve()
