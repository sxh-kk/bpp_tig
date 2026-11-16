"""Frozen LIBERO OSC actions -> TIG Panda absolute EE targets, without learning.

This module uses NumPy/SciPy only. Poses are xyz + quaternion WXYZ unless an
explicit order is supplied. Matrices T_A_B map coordinates in B into A.

Controller evidence (robosuite v1.4.0):
https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/config/osc_pose.json
https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/control_utils.py
OSC clips each input component to [-1, 1], then scales xyz by .05 m and the
axis-angle vector by .5 rad. Translation is world-aligned; rotation LEFT
multiplies the current world orientation. BPP's 6D rotation encodes that OSC
input vector as a rotation matrix, not an already-scaled physical rotation.

TIG pd_ee_pose takes absolute robot-base xyz + intrinsic XYZ Euler angles.
Its Panda gripper maps normalized [-1, 1] to finger target [-.01, .04] m;
positive opens. LIBERO's Panda command positive closes (negative opens).
The gripper bridge preserves this direction with full-open/full-close targets,
not robosuite's actuator-specific integrated speed/torque dynamics.

The default tool correspondence uses the common nominal Panda hand frame:
robosuite grip_site at hand-local z=.097 m with Rz(-pi/2), versus TIG
panda_hand_tcp at z=.1034 m with identity rotation. The .0064 m offset is a
site correspondence, not an assertion of identical robot FK across simulators.
https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/assets/grippers/panda_gripper.xml
https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/grippers/panda_gripper.py

Finite +/-2m base-coordinate bounds are numerical guards, not a claim that
Panda can reach that box. IK feasibility remains the simulator's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

OSC_OUTPUT_MAX = np.array([.05, .05, .05, .5, .5, .5], dtype=np.float64)
PANDA_GRIPPER_BOUNDS = (-.01, .04)
DEFAULT_WORKSPACE_BOUNDS = ((-2., -2., -2.), (2., 2., 2.))


def _array(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite with shape {shape}, got {value.shape}')
    return value.copy()


def quat_wxyz_to_xyzw(quaternion):
    return _array(quaternion, (4,), 'quaternion')[[1, 2, 3, 0]]


def quat_xyzw_to_wxyz(quaternion):
    return _array(quaternion, (4,), 'quaternion')[[3, 0, 1, 2]]


def pose_to_matrix(pose, *, quaternion_order='wxyz'):
    pose = _array(pose, (7,), 'pose')
    if quaternion_order not in ('wxyz', 'xyzw'):
        raise ValueError('quaternion_order must be wxyz or xyzw')
    quaternion = pose[3:]
    if quaternion_order == 'wxyz':
        quaternion = quat_wxyz_to_xyzw(quaternion)
    if np.linalg.norm(quaternion) < 1e-12:
        raise ValueError('zero quaternion does not define a pose')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def _transform(value, name='transform'):
    value = _array(value, (4, 4), name)
    rotation = value[:3, :3]
    if (not np.allclose(value[3], [0., 0., 0., 1.], atol=1e-8)
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError(f'{name} must be a proper rigid transform')
    return value


def matrix_to_pose(matrix, *, quaternion_order='wxyz'):
    matrix = _transform(matrix)
    if quaternion_order not in ('wxyz', 'xyzw'):
        raise ValueError('quaternion_order must be wxyz or xyzw')
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    if quaternion_order == 'wxyz':
        quaternion = quat_xyzw_to_wxyz(quaternion)
    return np.r_[matrix[:3, 3], quaternion]


def matrix_to_rot6d(matrix):
    """PyTorch3D/BPP convention: the first two ROWS, flattened row-major."""
    matrix = _array(matrix, (3, 3), 'rotation matrix')
    return matrix[:2].reshape(6).copy()


def rot6d_to_matrix(rotation):
    """Project two predicted rows by Gram-Schmidt; reject undefined rotations."""
    rotation = _array(rotation, (6,), 'rotation6d')
    first, second = rotation[:3], rotation[3:]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        raise ValueError('degenerate rotation6d: first row is zero')
    first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        raise ValueError('degenerate rotation6d: rows are collinear')
    second = second / second_norm
    return np.stack([first, second, np.cross(first, second)])


def default_tcp_to_libero_eef():
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('Z', -np.pi / 2).as_matrix()
    result[2, 3] = .097 - .1034
    return result


def panda_fingers_to_libero(finger_qpos):
    """Measured executing-robot proprio only; never demonstration prompt data."""
    fingers = _array(finger_qpos, (2,), 'finger_qpos')
    return fingers * np.array([1., -1.])


def panda_gripper_hold_command(finger_qpos):
    fingers = _array(finger_qpos, (2,), 'finger_qpos')
    low, high = PANDA_GRIPPER_BOUNDS
    raw = 2. * (float(np.mean(fingers)) - low) / (high - low) - 1.
    return float(np.clip(raw, -1., 1.)), bool(raw < -1. or raw > 1.)


@dataclass(frozen=True)
class ArmState:
    tcp_pose_world: np.ndarray
    base_pose_world: np.ndarray
    finger_qpos: np.ndarray

    def __post_init__(self):
        # Copy arrays so a simulator view cannot mutate a reset/hold reference.
        for key, shape in [('tcp_pose_world', (7,)), ('base_pose_world', (7,)),
                           ('finger_qpos', (2,))]:
            object.__setattr__(self, key, _array(getattr(self, key), shape, key))
        pose_to_matrix(self.tcp_pose_world)
        pose_to_matrix(self.base_pose_world)


@dataclass
class ControlStep:
    actions: dict[str, np.ndarray]
    diagnostics: dict


class ControlAdapter:
    def __init__(self, active_arm='panda_wristcam-0', *, policy_from_world=None,
                 tcp_to_libero_eef=None, workspace_bounds=DEFAULT_WORKSPACE_BOUNDS,
                 osc_output_max=OSC_OUTPUT_MAX):
        self.active_arm = active_arm
        self.policy_from_world = _transform(np.eye(4) if policy_from_world is None
                                            else policy_from_world, 'policy_from_world')
        self.tcp_to_libero_eef = _transform(default_tcp_to_libero_eef()
                                           if tcp_to_libero_eef is None else tcp_to_libero_eef,
                                           'tcp_to_libero_eef')
        self.workspace_bounds = _array(workspace_bounds, (2, 3), 'workspace_bounds')
        if np.any(self.workspace_bounds[0] >= self.workspace_bounds[1]):
            raise ValueError('workspace lower bounds must be below upper bounds')
        self.osc_output_max = _array(osc_output_max, (6,), 'osc_output_max')
        if np.any(self.osc_output_max <= 0):
            raise ValueError('OSC scaling must be strictly positive')
        self.clip_counts: dict[str, int] = {}
        self._holds: dict[str, tuple[np.ndarray, float]] = {}
        self._arm_names: set[str] = set()
        self.steps = 0

    def reset(self, states: Mapping[str, ArmState]):
        if self.active_arm not in states or len(states) != 2:
            raise ValueError('states must contain the selected arm and one inactive arm')
        self.clip_counts.clear()
        self._holds.clear()
        self._arm_names = set(states)
        self.steps = 0
        for name, state in states.items():
            if name != self.active_arm:
                hold, clipped = panda_gripper_hold_command(state.finger_qpos)
                self._holds[name] = (pose_to_matrix(state.tcp_pose_world), hold)
                if clipped:
                    self.clip_counts[f'{name}.initial_gripper'] = 1

    def proprioception(self, state: ArmState):
        eef = (self.policy_from_world @ pose_to_matrix(state.tcp_pose_world)
               @ self.tcp_to_libero_eef)
        return {'ee_pos': eef[:3, 3].astype(np.float32),
                'ee_ori': matrix_to_rot6d(eef[:3, :3]).astype(np.float32),
                'gripper_states': panda_fingers_to_libero(state.finger_qpos).astype(np.float32)}

    def _command(self, target_world, base_pose_world, gripper, name, events):
        base_target = np.linalg.inv(pose_to_matrix(base_pose_world)) @ target_world
        raw_position = base_target[:3, 3].copy()
        position = np.clip(raw_position, *self.workspace_bounds)
        events[f'{name}.position_guard'] = int(np.count_nonzero(position != raw_position))
        # ManiSkill's euler_angles_to_matrix(..., 'XYZ') means intrinsic XYZ.
        # At a gimbal singularity SciPy chooses a valid equivalent triple.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Gimbal lock detected.*', category=UserWarning)
            euler = Rotation.from_matrix(base_target[:3, :3]).as_euler('XYZ')
        command = np.r_[position, np.clip(euler, -np.pi, np.pi), gripper]
        if not np.isfinite(command).all():
            raise ValueError('nonfinite controller target')
        return command.astype(np.float32)

    def step(self, action10, states: Mapping[str, ArmState]):
        if not self._arm_names:
            self.reset(states)
        if set(states) != self._arm_names:
            raise ValueError('robot arm identities changed; reset the control adapter')
        action10 = _array(action10, (10,), 'BPP action10')
        rotation = rot6d_to_matrix(action10[3:9])
        raw = np.r_[action10[:3], Rotation.from_matrix(rotation).as_rotvec(), action10[9]]
        osc = np.clip(raw, -1., 1.)
        events = {'osc.position_input': int(np.count_nonzero(osc[:3] != raw[:3])),
                  'osc.rotation_input': int(np.count_nonzero(osc[3:6] != raw[3:6])),
                  'osc.gripper_input': int(osc[6] != raw[6])}
        physical_delta = osc[:6] * self.osc_output_max
        state = states[self.active_arm]
        current_policy_eef = (self.policy_from_world @ pose_to_matrix(state.tcp_pose_world)
                              @ self.tcp_to_libero_eef)
        target_policy_eef = current_policy_eef.copy()
        target_policy_eef[:3, 3] += physical_delta[:3]
        target_policy_eef[:3, :3] = (Rotation.from_rotvec(physical_delta[3:]).as_matrix()
                                    @ current_policy_eef[:3, :3])
        target_world_tcp = (np.linalg.inv(self.policy_from_world) @ target_policy_eef
                            @ np.linalg.inv(self.tcp_to_libero_eef))
        if osc[6] == 0.:
            gripper, clipped = panda_gripper_hold_command(state.finger_qpos)
            events['active.gripper_hold'] = int(clipped)
        else:
            gripper = float(-np.sign(osc[6]))
        actions = {self.active_arm: self._command(target_world_tcp, state.base_pose_world,
                                                  gripper, self.active_arm, events)}
        for name, (hold_world, hold_gripper) in self._holds.items():
            actions[name] = self._command(hold_world, states[name].base_pose_world,
                                           hold_gripper, name, events)
        for name, count in events.items():
            self.clip_counts[name] = self.clip_counts.get(name, 0) + count
        self.steps += 1
        return ControlStep(actions, {'step': self.steps, 'osc_input': osc.tolist(),
                                     'physical_delta': physical_delta.tolist(),
                                     'clip_events': events, 'clip_counts': self.clip_counts.copy(),
                                     'rotation6d_projected': not np.allclose(
                                         action10[3:9], matrix_to_rot6d(rotation), atol=1e-5)})
