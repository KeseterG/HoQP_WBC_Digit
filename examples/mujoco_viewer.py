from dm_control import mujoco
from mujoco import viewer
import os
import pinocchio as pin
from hoqp_wbc.wbc.torque_wbc import TorqueWBC, WBCSolution
from hoqp_wbc.utils.mj_pin import *

os.environ["MUJOCO_GL"] = "egl"
BASE_DIR = "/home/keseterg/Documents/Learning/robotics/HoQP_WBC_Digit/"

physics = mujoco.Physics.from_xml_path(
    os.path.join(BASE_DIR, "models", "digit", "digit-v3.xml")
)

robot = pin.RobotWrapper.BuildFromURDF(
    os.path.join(BASE_DIR, "models", "digit", "urdf", "digit_float.urdf"),
    package_dirs=[
        os.path.join(BASE_DIR, "models", "digit", "urdf")
    ],
    root_joint=pin.JointModelFreeFlyer()
)

wbc = TorqueWBC(robot.model, robot.data)
wbc.update_contact(
    contact_names=["toe_pitch_joint_left", "toe_pitch_joint_right"]
)




def update_sim():
    physics.step()
    physics.forward()


def update_wbc_control():
    wbc.update_measurement(
        mj_to_pin_q(physics.data.qpos),
        mj_to_pin_qd(physics.data.qvel),
        mj_to_pin_qd(physics.data.qacc_smooth),
    )
    sol: WBCSolution = wbc.solve()

    if sol is not None:
        physics.data.ctrl = pin_to_mj_tau(sol.torque)


with viewer.launch_passive(physics.model._model, physics.data._data) as v:
    # update functions
    while v.is_running():
        update_sim()
        update_wbc_control()
        v.sync()
