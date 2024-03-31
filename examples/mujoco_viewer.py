from copy import deepcopy

from dm_control import mujoco
from mujoco import viewer
import os
import pinocchio as pin
from qb_wbc.wbc.torque_wbc import TorqueWBC, WBCSolution
from qb_wbc.utils.mj_pin import *

os.environ["MUJOCO_GL"] = "egl"
BASE_DIR = "/home/keseterg/Documents/Learning/robotics/HoQP_WBC_Digit/"

physics = mujoco.Physics.from_xml_path(
    os.path.join(BASE_DIR, "models", "digit", "digit-v3-ungeared.xml")
)
physics.reset(0)
q_stand = deepcopy(physics.data.qpos)

robot = pin.RobotWrapper.BuildFromURDF(
    os.path.join(BASE_DIR, "models", "digit", "urdf", "digit_float.urdf"),
    package_dirs=[
        os.path.join(BASE_DIR, "models", "digit", "urdf")
    ],
    root_joint=pin.JointModelFreeFlyer()
)

# robot = robot.buildReducedRobot(
#     list_of_joints_to_lock=[
#         "toe_pitch_joint_left",
#         "toe_pitch_joint_right",
#         "toe_roll_joint_left",
#         "toe_roll_joint_right"
#     ],
#     reference_configuration=q_stand
# )
#

wbc = TorqueWBC(robot.model, robot.data)
wbc.update_contact(
    contact_names=["toe_pitch_joint_left", "toe_pitch_joint_right"]
)

UPDATE_HZ = 500
UPDATE_INTERVAL_S = 1 / UPDATE_HZ

q_stand_pin = mj_to_pin_q(q_stand)
kp = 500.0
k_act_id = [6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]  # rod, 9=10, 16=17
S_v2act = np.zeros((len(k_act_id), 29))  # actuator select matrix, mapping from qd to act
counter = 0
for col in k_act_id:
    S_v2act[counter][col + 1] = 1
    counter += 1

def run_control():
    wbc.update_measurement(
        mj_to_pin_q(physics.data.qpos),
        mj_to_pin_qd(physics.data.qvel),
        mj_to_pin_qd(physics.data.qacc),
    )
    sol: WBCSolution = wbc.solve()

    if sol is not None:
        # physics.data.ctrl = pin_to_mj_tau(sol.torque)
        err = (mj_to_pin_q(physics.data.qpos) - q_stand_pin)
        tau = -kp * S_v2act @ err
        physics.data.ctrl = pin_to_mj_tau(tau)


interval = 0.0
with viewer.launch_passive(physics.model._model, physics.data._data) as viewer:
    # update functions
    while viewer.is_running():
        physics.step()
        physics.forward()
        interval += physics.timestep()

        if interval > UPDATE_INTERVAL_S:
            interval = 0.0
            run_control()

        viewer.sync()
