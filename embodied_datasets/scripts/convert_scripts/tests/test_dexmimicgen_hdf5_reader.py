from pathlib import Path

import numpy as np
import pytest

from convert_core.errors import ConversionError
from convert_core.lerobot_writer import build_manifest
from readers.dexmimicgen_hdf5_reader import (
    PANDA_ACTION_KEYS,
    inspect_partition,
    iter_frames,
    simulator_state_names,
)


h5py = pytest.importorskip("h5py")


MODEL = """<mujoco><worldbody><body><joint name="robot0_joint1"/></body></worldbody></mujoco>"""


def _source(root: Path, *, episodes: int = 2, corrupt_actions: bool = False) -> Path:
    path = root / "generated" / "two_arm_threading.hdf5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        data.attrs["env_args"] = (
            '{"env_name":"TwoArmThreading","env_kwargs":'
            '{"env_lang":null,"robots":["Panda","Panda"],'
            '"camera_names":["agentview"],"control_freq":20}}'
        )
        total = 0
        for episode_index in range(episodes):
            frames = episode_index + 2
            total += frames
            demo = data.create_group(f"demo_{episode_index}")
            demo.attrs["num_samples"] = frames
            demo.attrs["model_file"] = MODEL
            action_dict = demo.create_group("action_dict")
            parts = []
            for key in PANDA_ACTION_KEYS:
                width = 1 if key.endswith("gripper") else 3
                values = (
                    np.arange(frames * width, dtype=np.float32).reshape(frames, width)
                    + len(parts)
                )
                action_dict.create_dataset(key, data=values)
                parts.append(values)
            actions = np.concatenate(parts, axis=1).astype(np.float64)
            if corrupt_actions:
                actions[0, 0] += 1
            demo.create_dataset("actions", data=actions)
            demo.create_dataset(
                "states", data=np.arange(frames * 3, dtype=np.float64).reshape(frames, 3)
            )
            obs = demo.create_group("obs")
            obs.create_dataset(
                "agentview_image", data=np.zeros((frames, 84, 84, 3), dtype=np.uint8)
            )
            obs.create_dataset(
                "robot0_joint_pos", data=np.arange(frames, dtype=np.float32).reshape(frames, 1)
            )
        data.attrs["total"] = total
    return path


def test_reader_preserves_schema_names_values_and_episode_provenance(tmp_path: Path):
    source = _source(tmp_path)
    info = inspect_partition(
        source,
        raw_dataset_root=tmp_path,
        collection_output=tmp_path / "stage",
        max_episodes=1,
    )
    assert info.partition_name == "threading"
    assert info.all_episode_count == 2
    assert info.all_frame_count == 5
    assert len(info.plan.episodes) == 1
    assert info.plan.episodes[0].extra["source_episode_id"] == "demo_0"
    assert info.plan.episodes[0].extra["source_model_sha256"]
    assert info.plan.feature_schema()["action"]["dtype"] == "float64"
    assert info.plan.feature_schema()["observation.sim_state"]["names"] == [
        "time",
        "qpos(robot0_joint1)",
        "qvel(robot0_joint1)",
    ]
    assert info.plan.feature_schema()["observation.robot0_joint_pos"]["names"] == [
        "robot0_joint1"
    ]
    frames = list(iter_frames(info.plan, info.plan.episodes[0]))
    assert len(frames) == 2
    assert frames[0]["action"].dtype == np.float64
    assert frames[0]["source.action_dict.right_rel_pos"].dtype == np.float32
    assert frames[0]["observation.images.agentview"].shape == (84, 84, 3)
    manifest = build_manifest(info.plan, reader_format="dexmimicgen_hdf5")
    assert manifest["episodes"][0]["source_episode_id"] == "demo_0"
    assert manifest["episodes"][0]["source_model_sha256"]
    assert manifest["action_semantics"]["actions_and_action_dict_both_preserved"] is True


def test_reader_rejects_actions_that_disagree_with_official_component_order(tmp_path: Path):
    source = _source(tmp_path, corrupt_actions=True)
    with pytest.raises(ConversionError, match="actions disagree with action_dict order"):
        inspect_partition(
            source,
            raw_dataset_root=tmp_path,
            collection_output=tmp_path / "stage",
        )


def test_simulator_state_names_use_mujoco_quaternion_order():
    free = '<mujoco><worldbody><body><joint name="payload" type="free"/></body></worldbody></mujoco>'
    names = simulator_state_names(free, expected_width=14, description="fixture")
    assert names[4:8] == (
        "qpos(payload).qw",
        "qpos(payload).qx",
        "qpos(payload).qy",
        "qpos(payload).qz",
    )


def test_reader_rejects_noncontinuous_demo_ids(tmp_path: Path):
    source = _source(tmp_path)
    with h5py.File(source, "r+") as output:
        output["data"].move("demo_1", "demo_2")
    with pytest.raises(ConversionError, match="not continuous"):
        inspect_partition(
            source,
            raw_dataset_root=tmp_path,
            collection_output=tmp_path / "stage",
        )
