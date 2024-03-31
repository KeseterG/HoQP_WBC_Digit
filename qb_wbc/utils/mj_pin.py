import numpy as np

postab_mj = """7
8
9
14
18
23
30
31
32
33
34
35
36
41
45
50
57
58
59
60
16
43""".split("\n")
postab_pin = """7
8
9
10
12
13
21
22
23
24
14
15
16
17
19
20
25
26
27
28
11
18""".split("\n")
veltab_mj = """6
7
8
12
16
20
26
27
28
29
30
31
32
36
40
44
50
51
52
53
14
38""".split("\n")
veltab_pin = """6
7
8
9
11
12
20
21
22
23
13
14
15
16
18
19
24
25
26
27
10
17""".split("\n")

pos_pin_mj_map = {}
pos_mj_pin_map = {}
vel_pin_mj_map = {}
vel_mj_pin_map = {}

for pmj, ppin in zip(postab_mj, postab_pin):
    pos_mj_pin_map[int(pmj)] = int(ppin)
    pos_pin_mj_map[int(ppin)] = int(pmj)

for vmj, vpin in zip(veltab_mj, veltab_pin):
    vel_mj_pin_map[int(vmj)] = int(vpin)
    vel_pin_mj_map[int(vpin)] = int(vmj)

gearing_mat = np.diag([80, 50, 16, 16, 50, 50, 80, 80, 50, 80, 80, 50, 16, 16, 50, 50, 80, 80, 50, 80])
def mj_to_pin_q(qpos: np.array) -> np.array:
    qpos_des = np.zeros(29)
    for k in pos_mj_pin_map.keys():
        qpos_des[pos_mj_pin_map[k]] = qpos[k]
    qpos_des[0:3] = qpos[0:3]
    qpos_des[3:6] = qpos[4:7]
    qpos_des[6] = qpos[3]

    return qpos_des


def mj_to_pin_qd(qvel: np.array) -> np.array:
    qvel_des = np.zeros(28)
    for k in vel_mj_pin_map.keys():
        qvel_des[vel_mj_pin_map[k]] = qvel[k]
    qvel_des[0:3] = qvel[0:3]
    qvel_des[3:6] = qvel[3:6]

    return qvel_des


def pin_to_mj_tau(tau: np.array) -> np.array:
    tau_des = np.zeros_like(tau)
    tau_des[0:6] = tau[0:6].copy()
    tau_des[6:10] = tau[12:16].copy()
    tau_des[10:16] = tau[6:12].copy()
    tau_des[16:20] = tau[16:20].copy()

    tau_des[4], tau_des[5] = (-0.5 * tau[4] + 1.5 * tau[5]),  (0.5 * tau[4] + 1.5 * tau[5])
    tau_des[14], tau_des[15] = (-0.5 * tau[14] + 1.5 * tau[15]),  (0.5 * tau[14] + 1.5 * tau[15])

    return tau_des
