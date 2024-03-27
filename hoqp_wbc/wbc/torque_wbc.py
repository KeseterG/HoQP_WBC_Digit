import dataclasses

import pinocchio as pin
from typing import List

from hoqp_wbc.qp.hoqp import HoQP
from hoqp_wbc.qp.task import Task
from hoqp_wbc.utils.common_types import *


@dataclasses.dataclass
class WBCSolution:
    qdd: Vector
    contact_forces: List[Vector]
    torque: Vector


class TorqueWBC:
    def __init__(self, model: PModel, data: PData):
        self.model: PModel = model
        self.data: PData = data

        # ------- Constants -------
        # Motion
        self.nq = self.model.nq  # joint angles dim
        self.nv = self.model.nv  # joint vel / acceleration dim
        self.k_passive_id = [0, 1, 2, 3, 4, 5, 10, 17]
        self.k_act_id = [6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]  # rod, 9=10, 16=17
        self.nact = len(self.k_act_id)
        self.S_v2act = np.zeros((self.nact, self.nv))  # actuator select matrix, mapping from qd to act
        counter = 0
        for col in self.k_act_id:
            self.S_v2act[counter][col] = 1
            counter += 1

        # Parameters
        self.k_tau_abs_max = 30.0
        self.k_mu = 0.5

        # contact
        self.friction_pyramid = np.array([
            [1, 0, -self.k_mu], [-1, 0, -self.k_mu],
            [0, 1, -self.k_mu], [0, -1, -self.k_mu]
        ])

        # ------- State Variables -------
        self.q = np.zeros(self.nq)
        self.qd = np.zeros(self.nv)
        self.qdd = np.zeros(self.nv)
        self.qdd_des = np.zeros(self.nv)
        self.contacts = []
        self.ncont = 0
        self.solve_count = 0

        # ------- State Matrices -------
        self.J_contact: Matrix = None
        self.Jd_contact: Matrix = None

        # ------- Control Targets -------
        self.des_x_centroid = 0


        # ------- QP Formulation -------
        # QP decision variables: [ qdd contact tau ]
        # QP decision variables dimensions: [ nv 3 * n_contact nact ]
        self.ndes = self.nv + len(self.contacts) + self.nact
        self.hoqp = HoQP(self.ndes)

        # ------- Tasks Formulation -------
        self._formulate_tasks()

    def _formulate_tasks(self):
        # ------- Tasks Definitions -------
        # Priority 0 Tasks:
        # -    Dynamics Constraints
        # -    Torque Limits
        # Priority 1 Tasks:
        # -    Optimal Control (Energy Optimization)
        self.hoqp.add_tasks(0, [
            self._formulate_floating_base_dynamics_task,
            self._formulate_contact_no_slip_task,
            self._formulate_friction_cone_task,
            # self._formulate_centroid_dynamics_task,
        ]),
        self.hoqp.add_tasks(1, [
            self._formulate_torque_limit_task,
            self._formulate_optimal_control_task
        ])

    def update_measurement(self, q: Vector, qd: Vector, qdd: Vector):
        self.q = q
        self.qd = qd
        self.qdd = qdd
        self._update_system_status()

    def update_contact(self, contact_ids: List[int] = None, contact_names: List[str] = None):
        if contact_ids:
            self.contacts = []
            for contact_ee_id in contact_ids:
                assert 7 <= contact_ee_id <= self.model.nq
                self.contacts.append(contact_ee_id)
        elif contact_names:
            self.contacts = []
            for contact_name in contact_names:
                frame_id = self.model.getFrameId(contact_name)
                assert frame_id != -1
                self.contacts.append(frame_id)
        self.ncont = len(self.contacts)

    def solve(self) -> WBCSolution:
        x = self.hoqp.solve()
        return WBCSolution(
            qdd=x[0:self.nv],
            contact_forces=[x[self.nv + i * 3:self.nv + (i + 1) * 3] for i in range(self.ncont)],
            torque=x[self.nv + 3 * self.ncont:]
        ) if x is not None else None

    def _update_system_status(self) -> None:
        # kinematics
        pin.forwardKinematics(self.model, self.data, self.q, self.qd)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, self.q, self.qd)

        # dynamics
        pin.crba(self.model, self.data, self.q)
        pin.nonLinearEffects(self.model, self.data, self.q, self.qd)

        # centroid model
        pin.computeTotalMass(self.model)
        pin.computeCentroidalMap(self.model, self.data, self.q)
        pin.computeCentroidalMapTimeVariation(self.model, self.data, self.q, self.qd)

        # contacts
        self.J_contact = np.vstack([
            pin.getFrameJacobian(self.model, self.data, contact_id, pin.LOCAL_WORLD_ALIGNED)[0:3, :]
            for contact_id in self.contacts
        ])  # contact jacobian, only take top tree rows for constraint linear forces
        self.Jd_contact = np.vstack([
            pin.getFrameJacobianTimeVariation(self.model, self.data, contact_id, pin.LOCAL_WORLD_ALIGNED)[0:3, :]
            for contact_id in self.contacts
        ])

        # other status
        self.ndes = self.nv + 3 * self.ncont + self.nact
        self.hoqp.update_n_decision(self.ndes)

    def _formulate_floating_base_dynamics_task(self) -> Task:
        # ensure dynamics equation is obeyed.
        # M @ qdd + nle = J.T @ lambda + S.T @ tau
        task = Task.EMPTY(self.ndes)
        task.A = np.hstack([
            self.data.M, -self.J_contact.T, -self.S_v2act.T
        ])
        task.b = -self.data.nle

        return task

    def _formulate_contact_no_slip_task(self) -> Task:
        # ensure the point of contact has no slip.
        # J @ qdd + Jd @ qd = 0 <-> xdd = 0
        task = Task.EMPTY(self.ndes)

        padding = np.zeros((6, self.ncont * 3 + self.nact))
        task.A = np.hstack([self.J_contact, padding])
        task.b = self.Jd_contact @ self.qd

        return task

    def _formulate_lock_accel_task(self) -> Task:
        # make joint acceleration reach desired values.
        # solve at low priority to ensure that higher priorities, such as dynamics and no slip, is obeyed.
        task = Task.EMPTY(self.ndes)

        padding = np.zeros((self.nv, self.ncont * 3 + self.nact))
        task.A = np.hstack([np.eye(self.nv), padding])
        task.b = np.zeros(self.nv)

        return task

    def _formulate_friction_cone_task(self) -> Task:
        task = Task.EMPTY(self.ndes)

        task.D = np.zeros((4 * self.ncont, self.ndes))
        for i in range(self.ncont):
            task.D[i * 4:(i + 1) * 4, self.nv + i*3:self.nv + (i + 1) * 3] = self.friction_pyramid.copy()
        task.f = np.zeros(4 * self.ncont)

        return task

    def _formulate_centroid_dynamics_task(self) -> Task:
        task = Task.EMPTY(self.ndes)

        l_g = self.data.Ag @ self.qd
        l_gd = (self.data.dAg @ self.qd + self.data.Ag @ self.qdd)[3:]
        M = self.data.M

        padding = np.zeros((3, self.nv - 3 + 3 * self.ncont + self.nact))
        task.A = np.hstack([np.eye(3), padding])
        task.b = l_gd

        return task

    def _formulate_torque_limit_task(self) -> Task:
        # ensure torque is feasible.
        # I @ tau <= tau_max
        # -I @ tau <= -tau_min
        task = Task.EMPTY(self.ndes)
        padding = np.zeros((self.nact * 2, self.nv + 3 * self.ncont))
        S_act2act = np.eye(self.nact)
        task.D = np.hstack([
            padding, np.vstack([S_act2act, -S_act2act])
        ])
        task.f = np.ones(self.nact * 2) * self.k_tau_abs_max

        return task

    def _formulate_optimal_control_task(self) -> Task:
        # optimal control, minimize the torque.
        # solve at low priority.
        task = Task.EMPTY(self.ndes)
        task.A = np.hstack([np.zeros((self.nact, self.nv + self.ncont * 3)), np.eye(self.nact)])
        task.b = np.zeros(self.nact)

        return task
