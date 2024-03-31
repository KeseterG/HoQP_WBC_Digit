import numpy as np
from copy import copy, deepcopy
from qb_wbc.qp import wqp, task

def test_wqp():
    task1 = task.Task(
        A=np.random.rand(2, 4),
        b=np.ones(2),
        D=np.random.rand(2, 4),
        f=np.ones(2)
    )

    def t1():
        return task1

    w = wqp.WQP(4)
    w.add_weighted_task(t1, 1.5)
    w.solve()
    x_1 = w.get_solution()

    eps = 1e-5

    task1.is_fit(x_1, eps)
