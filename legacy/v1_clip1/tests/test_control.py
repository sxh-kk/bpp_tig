"""Checks for simulator-independent unit/frame/sign mistakes in control bridging."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from examples.baselines.bpp.bpp.control import (
    ArmState, ControlAdapter, default_tcp_to_libero_eef, matrix_to_pose,
    matrix_to_rot6d, panda_fingers_to_libero, panda_gripper_hold_command,
    pose_to_matrix, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, rot6d_to_matrix,
)


def pose(position=(0., 0., 0.), rotation=None):
    matrix = np.eye(4)
    matrix[:3, 3] = position
    if rotation is not None:
        matrix[:3, :3] = rotation
    return matrix_to_pose(matrix)


def action(position=(0., 0., 0.), rotation_vector=(0., 0., 0.), gripper=0.):
    rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    return np.r_[position, matrix_to_rot6d(rotation), gripper]


def states():
    return {'panda_wristcam-0': ArmState(pose((.4, .1, .5)), pose(), [.02, .02]),
            'panda_wristcam-1': ArmState(pose((-.3, .2, .4)), pose(), [.04, .04])}


def command_pose(command):
    matrix = np.eye(4)
    matrix[:3, 3] = command[:3]
    matrix[:3, :3] = Rotation.from_euler('XYZ', command[3:6]).as_matrix()
    return matrix


class GeometryTests(unittest.TestCase):
    def test_rotation6d_is_rows_not_columns(self):
        expected = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        encoded = np.array([0., -1., 0., 1., 0., 0.])
        assert_allclose(rot6d_to_matrix(encoded), expected, atol=1e-12)
        assert_allclose(matrix_to_rot6d(expected), encoded)

    def test_rotation6d_repairs_nonorthogonal_prediction(self):
        recovered = rot6d_to_matrix([2., 0., 0., .2, 3., 0.])
        assert_allclose(recovered, np.eye(3), atol=1e-12)
        with self.assertRaises(ValueError):
            rot6d_to_matrix([1., 0., 0., 2., 0., 0.])

    def test_quaternion_orders_and_nonidentity_pose(self):
        expected = pose((1., -.2, .3), Rotation.from_euler('XYZ', [.3, -.8, .5]).as_matrix())
        xyzw = np.r_[expected[:3], quat_wxyz_to_xyzw(expected[3:])]
        assert_allclose(pose_to_matrix(expected), pose_to_matrix(xyzw, quaternion_order='xyzw'))
        assert_allclose(quat_xyzw_to_wxyz(xyzw[3:]), expected[3:])
        assert_allclose(pose_to_matrix(matrix_to_pose(pose_to_matrix(expected))), pose_to_matrix(expected))
        with self.assertRaises(ValueError):
            pose_to_matrix(np.zeros(7))

    def test_tool_correspondence_and_finger_signs(self):
        tool = default_tcp_to_libero_eef()
        assert_allclose(tool[:3, 3], [0., 0., -.0064], atol=1e-12)
        assert_allclose(tool[:3, :3], [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], atol=1e-12)
        assert_allclose(panda_fingers_to_libero([.02, .03]), [.02, -.03])
        self.assertAlmostEqual(panda_gripper_hold_command([0., 0.])[0], -.6)
        self.assertAlmostEqual(panda_gripper_hold_command([.04, .04])[0], 1.)

    def test_proprio_uses_same_eef_correspondence_as_control(self):
        data = states()
        adapter = ControlAdapter()
        proprio = adapter.proprioception(data[adapter.active_arm])
        assert_allclose(proprio['ee_pos'], [.4, .1, .4936], atol=1e-7)
        assert_allclose(rot6d_to_matrix(proprio['ee_ori']), default_tcp_to_libero_eef()[:3, :3], atol=1e-7)
        assert_allclose(proprio['gripper_states'], [.02, -.02], atol=1e-7)


class ControllerTests(unittest.TestCase):
    def test_zero_action_holds_both_arm_poses_and_current_gripper(self):
        data = states()
        result = ControlAdapter().step(action(), data)
        for name, state in data.items():
            assert_allclose(command_pose(result.actions[name]), pose_to_matrix(state.tcp_pose_world), atol=1e-7)
        self.assertAlmostEqual(result.actions['panda_wristcam-0'][6], .2, places=6)
        self.assertEqual(result.actions['panda_wristcam-1'][6], 1.)

    def test_osc_translation_is_scaled_and_world_aligned(self):
        data = states()
        data['panda_wristcam-0'] = ArmState(
            pose((.4, .1, .5), Rotation.from_euler('Z', np.pi / 2).as_matrix()), pose(), [.02, .02])
        result = ControlAdapter().step(action(position=(1., 0., 0.)), data)
        # A body-frame implementation would move y, or direct-meters would move 1m.
        assert_allclose(result.actions['panda_wristcam-0'][:3], [.45, .1, .5], atol=1e-7)

    def test_osc_rotation_is_scaled_and_left_multiplied(self):
        data = states()
        current_tcp = Rotation.from_euler('XYZ', [.7, -.2, .4]).as_matrix()
        data['panda_wristcam-0'] = ArmState(pose((.4, .1, .5), current_tcp), pose(), [.02, .02])
        result = ControlAdapter().step(action(rotation_vector=(1., 0., 0.)), data)
        expected = Rotation.from_rotvec([.5, 0., 0.]).as_matrix() @ current_tcp
        assert_allclose(command_pose(result.actions['panda_wristcam-0'])[:3, :3], expected, atol=1e-7)

    def test_rotation_pivots_about_libero_grip_site_not_tcp(self):
        data = states()
        result = ControlAdapter().step(action(rotation_vector=(1., 0., 0.)), data)
        before = pose_to_matrix(data['panda_wristcam-0'].tcp_pose_world) @ default_tcp_to_libero_eef()
        after = command_pose(result.actions['panda_wristcam-0']) @ default_tcp_to_libero_eef()
        assert_allclose(after[:3, 3], before[:3, 3], atol=1e-7)
        self.assertGreater(abs(result.actions['panda_wristcam-0'][1] - .1), .001)

    def test_world_goal_is_converted_to_rotated_robot_base(self):
        data = states()
        base = pose((1., .5, -.1), Rotation.from_euler('Z', np.pi / 2).as_matrix())
        data['panda_wristcam-0'] = ArmState(data['panda_wristcam-0'].tcp_pose_world, base, [.02, .02])
        result = ControlAdapter().step(action(position=(1., 0., 0.)), data)
        expected_world = pose_to_matrix(data['panda_wristcam-0'].tcp_pose_world)
        expected_world[0, 3] += .05
        reconstructed_world = pose_to_matrix(base) @ command_pose(result.actions['panda_wristcam-0'])
        assert_allclose(reconstructed_world, expected_world, atol=1e-7)

    def test_nonidentity_policy_frame_transforms_translation(self):
        policy_from_world = np.eye(4)
        policy_from_world[:3, :3] = Rotation.from_euler('Z', np.pi / 2).as_matrix()
        result = ControlAdapter(policy_from_world=policy_from_world).step(action(position=(1., 0., 0.)), states())
        assert_allclose(result.actions['panda_wristcam-0'][:3], [.4, .05, .5], atol=1e-7)

    def test_libero_close_open_sign_and_inactive_hold_survive_drift(self):
        data = states()
        adapter = ControlAdapter()
        adapter.reset(data)
        self.assertEqual(adapter.step(action(gripper=1.), data).actions['panda_wristcam-0'][6], -1.)
        self.assertEqual(adapter.step(action(gripper=-1.), data).actions['panda_wristcam-0'][6], 1.)
        data['panda_wristcam-1'] = ArmState(pose((-.1, .1, .8)), pose(), [0., 0.])
        result = adapter.step(action(), data)
        assert_allclose(result.actions['panda_wristcam-1'][:3], [-.3, .2, .4], atol=1e-7)
        self.assertEqual(result.actions['panda_wristcam-1'][6], 1.)
        adapter.reset(data)
        assert_allclose(adapter.step(action(), data).actions['panda_wristcam-1'][:3], [-.1, .1, .8], atol=1e-7)

    def test_finite_input_and_workspace_clipping_are_reported(self):
        adapter = ControlAdapter(workspace_bounds=([-.42, -.2, 0.], [.42, .2, 1.]))
        result = adapter.step(action(position=(2., 0., 0.), rotation_vector=(1.3, 0., 0.), gripper=2.), states())
        self.assertAlmostEqual(result.actions['panda_wristcam-0'][0], .42, places=6)
        self.assertEqual(result.diagnostics['clip_events']['osc.position_input'], 1)
        self.assertEqual(result.diagnostics['clip_events']['osc.rotation_input'], 1)
        self.assertEqual(result.diagnostics['clip_events']['osc.gripper_input'], 1)
        self.assertEqual(result.diagnostics['clip_events']['panda_wristcam-0.position_guard'], 1)
        adapter.reset(states())
        self.assertEqual(adapter.steps, 0)
        with self.assertRaises(ValueError):
            adapter.step(np.full(10, np.nan), states())
        with self.assertRaises(ValueError):
            ControlAdapter(workspace_bounds=([0., 0., 0.], [np.inf, 1., 1.]))

    def test_intrinsic_xyz_round_trip_at_gimbal_singularity(self):
        data = states()
        rotation = Rotation.from_euler('XYZ', [.4, np.pi / 2, -.2]).as_matrix()
        data['panda_wristcam-0'] = ArmState(pose((.4, .1, .5), rotation), pose(), [.02, .02])
        result = ControlAdapter().step(action(), data)
        assert_allclose(command_pose(result.actions['panda_wristcam-0'])[:3, :3], rotation, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
