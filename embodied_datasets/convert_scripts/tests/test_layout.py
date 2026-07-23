import pytest

pytest.importorskip("lerobot")

from pathlib import Path

import numpy as np

from common_convert.layout import assemble_state

URDF_PATH = str(Path(__file__).parent / "fixtures" / "simple_arm.urdf")


def test_assemble_state_without_urdf_zero_fills_eef():
    joints = np.zeros((3, 2))
    gripper = np.ones((3, 1))
    state = assemble_state(joints, gripper, urdf_path=None, dof_per_arm=2)
    assert state.shape == (3, 2 + 3 + 4 + 1)
    assert np.array_equal(state[:, 2:9], np.zeros((3, 7)))
    assert np.array_equal(state[:, 9:], gripper)


def test_assemble_state_with_urdf_computes_fk_eef_pos():
    joints = np.zeros((2, 2))
    gripper = np.zeros((2, 1))
    state = assemble_state(joints, gripper, urdf_path=URDF_PATH, dof_per_arm=2)
    assert np.allclose(state[:, 2:4], [1.5, 0.0], atol=1e-6)


def test_assemble_state_column_order_is_joint_eef_pos_eef_quat_gripper():
    joints = np.array([[0.1, 0.2]])
    gripper = np.array([[0.9, 0.8]])
    state = assemble_state(joints, gripper, urdf_path=None, dof_per_arm=2)
    assert np.array_equal(state[:, :2], joints)
    assert np.array_equal(state[:, 9:], gripper)
