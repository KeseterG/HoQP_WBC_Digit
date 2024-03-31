import numpy as np
from copy import copy, deepcopy
from qb_wbc.qp import hoqp

def test_hoqp_level():
    task1 = hoqp.Task(
        A=np.random.rand(2, 4),
        b=np.ones(2),
        D=np.random.rand(2, 4),
        f=np.ones(2)
    )
    task2 = deepcopy(task1)
    task2.A = np.ones((2, 4))

    level1: hoqp.HoQPLevel = hoqp.HoQPLevel(task1, None)
    level2: hoqp.HoQPLevel = hoqp.HoQPLevel(task2, level1)

    level1.solve()
    level2.solve()

    x_1 = level1.get_solution()
    x_2 = level2.get_solution()
    v_1 = level1.v_p_star
    v_2 = level2.v_p_star

    eps = 1e-5

    print(task1.get_fitness(x_1))
    print(task2.get_fitness(x_2))

    assert task1.is_fit(x_1, eps)
    assert task2.is_fit(x_2, eps)

def test_hoqp_solver():
    task1 = hoqp.Task(
        A=np.random.rand(2, 4),
        b=np.ones(2),
        D=np.random.rand(2, 4),
        f=np.ones(2)
    )
    task2 = deepcopy(task1)
    task2.A = np.ones((2, 4))

    def t1():
        return task1
    def t2():
        return task2

    h = hoqp.HoQP(3)
    h.update_n_decision(4)
    h.add_task(0, t1)
    h.add_task(1, t2)
    x = h.solve()

    assert x is not None
    eps = 1e-5
    assert task1.is_fit(x, eps)
    assert task2.is_fit(x, eps)

