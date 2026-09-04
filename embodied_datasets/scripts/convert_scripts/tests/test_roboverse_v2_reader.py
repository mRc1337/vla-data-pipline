from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from convert_core.errors import ConversionError
from readers.roboverse_v2_reader import (
    action_columns,
    column_array,
    inspect_collection,
    iter_part_frames,
    jsonable_static_payload,
    state_columns,
)


def _entity(robot: str = "franka", offset: float = 0.0) -> dict:
    return {
        robot: {
            "pos": [offset, 0.0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
            "dof_pos": {"joint_a": offset, "joint_b": offset + 1.0},
        }
    }


def _episode(*, actions: int = 2, states: int | None = 2, array_action: bool = False) -> dict:
    if array_action:
        action_values = [np.asarray([index, index + 1], dtype=np.float64) for index in range(actions)]
    else:
        action_values = [
            {"dof_pos_target": {"joint_a": float(index), "joint_b": float(index + 1)}}
            for index in range(actions)
        ]
    return {
        "actions": action_values,
        "states": None if states is None else [_entity(offset=float(index)) for index in range(states)],
        "init_state": _entity(),
        "extra": {"task_name": "official_task", "seed": np.int32(7)},
    }


def _write_source(root: Path, name: str, payload: dict, *, compressed: bool = False) -> Path:
    path = root / "trajs" / "suite" / "task" / "v2" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compressed else open
    with opener(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


def test_aligned_episode_keeps_named_float64_action_and_state_fields(tmp_path: Path):
    path = _write_source(tmp_path, "robot_v2.pkl", {"franka": [_episode()]})

    plan = inspect_collection(tmp_path)

    assert plan.num_source_episodes == 1
    assert len(plan.parts) == 1
    part = plan.parts[0]
    assert part.stream_kind == "aligned"
    assert part.robot_name == "franka"
    assert part.num_frames == 2
    assert [column.feature_key for column in part.feature_columns] == [
        "action",
        "observation.state.franka.pos",
        "observation.state.franka.rot",
        "observation.state.franka.dof_pos",
    ]
    assert part.feature_columns[0].names == ("joint_a", "joint_b")
    assert part.feature_columns[0].dtype == "float64"
    assert part.feature_statistics[0].frames == 2
    assert part.feature_statistics[0].finite_min == (0.0, 1.0)
    assert part.feature_statistics[0].finite_max == (1.0, 2.0)
    assert part.feature_statistics[0].nan_count == (0, 0)
    assert part.episodes[0].task_origin == "episode.extra.task_name"

    frames = list(iter_part_frames(part, part.episodes[0]))
    assert len(frames) == 2
    assert frames[0]["task"] == "official_task"
    np.testing.assert_array_equal(frames[1]["action"], [1.0, 2.0])
    assert part.episodes[0].source_file.path == path


def test_multidimensional_numeric_state_preserves_exact_shape(tmp_path: Path):
    episode = _episode()
    episode["states"][0]["franka"]["dof_pos"] = {
        "joint_a": np.asarray([0.0], dtype=np.float64),
        "joint_b": np.asarray([1.0], dtype=np.float64),
    }
    episode["states"][1]["franka"]["dof_pos"] = {
        "joint_a": np.asarray([2.0], dtype=np.float64),
        "joint_b": np.asarray([3.0], dtype=np.float64),
    }
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [episode]})

    part = inspect_collection(tmp_path).parts[0]
    column = next(
        item for item in part.feature_columns
        if item.feature_key == "observation.state.franka.dof_pos"
    )

    assert column.shape == (2, 1)
    assert column.dtype == "float64"
    assert column.names == ("joint_a", "joint_b")
    assert column.source_component_shapes == ((1,), (1,))
    frames = list(iter_part_frames(part, part.episodes[0]))
    np.testing.assert_array_equal(
        frames[1]["observation.state.franka.dof_pos"],
        np.asarray([[2.0], [3.0]], dtype=np.float64),
    )


def test_mismatched_action_state_lengths_become_linkable_separate_streams(tmp_path: Path):
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [_episode(actions=3, states=2)]})

    plan = inspect_collection(tmp_path)

    assert [(part.stream_kind, part.num_frames) for part in plan.parts] == [
        ("action", 3),
        ("state", 2),
    ]
    assert plan.num_source_episodes == 1
    assert plan.num_output_episodes == 2


def test_missing_states_stays_action_only_without_fill(tmp_path: Path):
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [_episode(states=None)]})

    plan = inspect_collection(tmp_path)

    assert len(plan.parts) == 1
    assert plan.parts[0].stream_kind == "action"
    assert all(column.feature_key.startswith("action") for column in plan.parts[0].feature_columns)


def test_array_action_names_come_from_source_robot_dof_order(tmp_path: Path):
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [_episode(array_action=True)]})

    plan = inspect_collection(tmp_path)

    action = next(column for column in plan.parts[0].feature_columns if column.feature_key == "action")
    assert action.names == ("joint_a", "joint_b")
    assert action.dtype == "float64"


def test_compressed_duplicate_alias_is_not_converted_twice(tmp_path: Path):
    payload = {"franka": [_episode()]}
    _write_source(tmp_path, "robot_v2.pkl", payload)
    _write_source(tmp_path, "robot_v2.pkl.gz", payload, compressed=True)

    plan = inspect_collection(tmp_path)

    assert plan.num_source_episodes == 1
    assert plan.duplicate_aliases == (("trajs/suite/task/v2/robot_v2.pkl.gz", "trajs/suite/task/v2/robot_v2.pkl"),)


def test_initial_state_json_is_reported_as_sidecar_not_episode(tmp_path: Path):
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [_episode()]})
    sidecar = tmp_path / "trajs" / "suite" / "initial_state_v2.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text('{"object": {"pos": [0, 0, 0]}}', encoding="utf-8")

    plan = inspect_collection(tmp_path)

    assert plan.sidecars == ("trajs/suite/initial_state_v2.json",)
    assert plan.sidecar_payloads == (
        {
            "relative_path": "trajs/suite/initial_state_v2.json",
            "payload": {"object": {"pos": [0, 0, 0]}},
        },
    )


def test_robot_enveloped_static_json_is_type_preserved_as_sidecar(tmp_path: Path):
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [_episode()]})
    sidecar = tmp_path / "trajs" / "suite" / "debug" / "franka_v2.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {"franka": [{"init_state": {"franka": {"dof_pos": {"joint_a": 0.0}}}}]}
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    plan = inspect_collection(tmp_path)

    assert plan.sidecars == ("trajs/suite/debug/franka_v2.json",)
    assert plan.sidecar_payloads == (
        {
            "relative_path": "trajs/suite/debug/franka_v2.json",
            "payload": payload,
        },
    )
    assert plan.source_issues == ()


def test_unreadable_trajectory_is_reported_as_blocking_while_scan_continues(tmp_path: Path):
    _write_source(tmp_path, "good_v2.pkl", {"franka": [_episode()]})
    broken = tmp_path / "trajs" / "suite" / "task" / "v2" / "broken_v2.pkl"
    broken.touch()

    plan = inspect_collection(tmp_path)

    assert plan.num_source_episodes == 1
    assert plan.source_issues == (
        {
            "relative_path": "trajs/suite/task/v2/broken_v2.pkl",
            "kind": "unreadable_or_invalid_trajectory",
            "detail": f"cannot read RoboVerse source {broken}: Ran out of input",
            "blocking": True,
            "local_size": 0,
        },
    )


def test_schema_change_inside_episode_fails_instead_of_casting_or_filling(tmp_path: Path):
    episode = _episode()
    episode["actions"][1]["dof_pos_target"].pop("joint_b")
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [episode]})

    with pytest.raises(ConversionError, match="schema changes at frame 1"):
        inspect_collection(tmp_path)


def test_opt_in_promotes_dynamic_integer_component_exactly_and_records_runs(
    tmp_path: Path,
):
    episode = _episode(states=None)
    episode["actions"] = [
        {
            "dof_pos_target": {
                "arm": np.float32(0.25),
                "finger": np.float32(1.0),
            }
        },
        {
            "dof_pos_target": {
                "arm": np.float32(0.5),
                "finger": 0,
            }
        },
        {
            "dof_pos_target": {
                "arm": np.float32(0.75),
                "finger": 1,
            }
        },
    ]
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [episode]})

    plan = inspect_collection(tmp_path, allow_lossless_dtype_promotion=True)

    part = plan.parts[0]
    assert [(column.feature_key, column.dtype) for column in part.feature_columns] == [
        ("action", "float32")
    ]
    assert part.feature_columns[0].source_dtype_options == (
        ("float32",),
        ("float32", "int64"),
    )
    promotion = part.episodes[0].action_dtype_promotions[0]
    assert promotion.source_component == "finger"
    assert promotion.source_component_index == 1
    assert promotion.target_dtype == "float32"
    assert [(run.start, run.end_exclusive, run.dtype) for run in promotion.runs] == [
        (0, 1, "float32"),
        (1, 3, "int64"),
    ]
    frames = list(iter_part_frames(part, part.episodes[0]))
    assert all(frame["action"].dtype == np.dtype("float32") for frame in frames)
    np.testing.assert_array_equal(frames[1]["action"], np.asarray([0.5, 0.0], dtype=np.float32))


def test_opt_in_rejects_integer_not_exactly_representable_in_existing_float(
    tmp_path: Path,
):
    episode = _episode(states=None)
    episode["actions"] = [
        {"dof_pos_target": {"finger": np.float32(0.0)}},
        {"dof_pos_target": {"finger": 16_777_217}},
    ]
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [episode]})

    with pytest.raises(ConversionError, match="cannot be promoted losslessly"):
        inspect_collection(tmp_path, allow_lossless_dtype_promotion=True)


def test_max_episodes_stops_parsing_inside_source_file(tmp_path: Path):
    invalid = _episode()
    invalid["actions"][1]["dof_pos_target"].pop("joint_b")
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [_episode(), invalid]})

    plan = inspect_collection(tmp_path, max_episodes=1)

    assert plan.num_source_episodes == 1
    assert plan.parts[0].num_frames == 2


def test_task_selection_is_applied_before_episode_limit(tmp_path: Path):
    first = _episode()
    first["extra"]["task_name"] = "first_task"
    second = _episode()
    second["extra"]["task_name"] = "selected_task"
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [first, second]})

    plan = inspect_collection(tmp_path, tasks={"selected_task"}, max_episodes=1)

    assert plan.num_source_episodes == 1
    selected = plan.parts[0].episodes[0]
    assert selected.source_episode_index == 1
    assert selected.source_task == "selected_task"


def test_source_statistics_distinguish_finite_and_nonfinite_values(tmp_path: Path):
    episode = _episode(states=None)
    episode["actions"] = [
        {"dof_pos_target": {"joint_a": float("nan"), "joint_b": float("inf")}},
        {"dof_pos_target": {"joint_a": float("-inf"), "joint_b": 3.0}},
    ]
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [episode]})

    statistics = inspect_collection(tmp_path).parts[0].feature_statistics[0]

    assert statistics.frames == 2
    assert statistics.finite_min == (None, 3.0)
    assert statistics.finite_max == (None, 3.0)
    assert statistics.nan_count == (1, 0)
    assert statistics.positive_infinity_count == (0, 1)
    assert statistics.negative_infinity_count == (1, 0)


def test_static_payload_preserves_numpy_and_nonfinite_type_information():
    encoded = jsonable_static_payload(
        {"scalar": np.float32(1.25), "vector": np.asarray([1, 2], dtype=np.int16), "nan": float("nan")}
    )

    assert encoded["scalar"] == {"__scalar__": 1.25, "dtype": "float32"}
    assert encoded["vector"] == {"__array__": [1, 2], "dtype": "int16", "shape": [2]}
    assert encoded["nan"] == {"__float__": "nan"}


def test_column_array_refuses_dtype_change():
    from readers.roboverse_v2_reader import FeatureColumn

    column = FeatureColumn("action", ("target",), "float64", (1,), ("joint",))
    with pytest.raises(ConversionError, match="changed from"):
        column_array({"target": np.asarray([1], dtype=np.float32)}, column)


def test_unnamed_source_vector_does_not_receive_placeholder_names():
    columns = state_columns({"entity": {"source_specific_vector": [1.0, 2.0]}})

    assert columns[0].feature_key == "observation.state.entity.source_specific_vector"
    assert columns[0].names is None


def test_action_key_normalization_collision_is_rejected():
    with pytest.raises(ConversionError, match="feature-key collision"):
        action_columns({"a-b": [1.0], "a_b": [2.0]}, {}, "robot")


def test_empty_state_mapping_is_preserved_as_schema_provenance(tmp_path: Path):
    episode = _episode()
    for state in episode["states"]:
        state["rigid_object"] = {"pos": [0.0, 0.0, 0.0], "dof_pos": {}}
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [episode]})

    part = inspect_collection(tmp_path).parts[0]

    assert part.empty_state_fields == (("rigid_object", "dof_pos", "empty_mapping"),)
    assert all("rigid_object.dof_pos" not in column.feature_key for column in part.feature_columns)


def test_empty_state_field_schema_change_is_rejected(tmp_path: Path):
    episode = _episode()
    episode["states"][0]["rigid_object"] = {"dof_pos": {}}
    episode["states"][1]["rigid_object"] = {"dof_pos": None}
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [episode]})

    with pytest.raises(ConversionError, match="empty state-field schema changes"):
        inspect_collection(tmp_path)


def test_non_robot_root_payload_is_preserved_with_episode_provenance(tmp_path: Path):
    _write_source(
        tmp_path,
        "robot_v2.pkl",
        {
            "metadata": {"task_name": "root task", "rate": np.int16(3)},
            "release_note": "kept",
            "franka": [_episode()],
        },
    )

    episode = inspect_collection(tmp_path).parts[0].episodes[0]

    assert episode.task_text == "official_task"
    assert episode.static_payload["source_file_payload"] == {
        "metadata": {
            "rate": {"__scalar__": 3, "dtype": "int16"},
            "task_name": "root task",
        },
        "release_note": "kept",
    }


def test_non_trajectory_file_is_inventoried_as_unconsumed_auxiliary(tmp_path: Path):
    _write_source(tmp_path, "robot_v2.pkl", {"franka": [_episode()]})
    auxiliary = tmp_path / "trajs" / "suite" / "notes.txt"
    auxiliary.write_text("evidence", encoding="utf-8")

    plan = inspect_collection(tmp_path)

    assert plan.auxiliary_files == (
        {
            "relative_path": "trajs/suite/notes.txt",
            "size": 8,
            "mtime_ns": auxiliary.stat().st_mtime_ns,
            "role": "not_consumed_by_trajectory_converter",
        },
    )


def test_stable_mixed_dtype_named_action_is_split_without_casting():
    step = {"target": {"joint": np.float32(1.25), "gripper": 1}}

    columns = action_columns(step, {}, "robot")

    assert [(column.feature_key, column.dtype, column.names) for column in columns] == [
        ("action.joint", "float32", ("joint",)),
        ("action.gripper", "int64", ("gripper",)),
    ]
    assert [column.source_component_indices for column in columns] == [(0,), (1,)]
    assert np.array_equal(column_array(step, columns[0]), np.asarray([1.25], dtype=np.float32))
    assert np.array_equal(column_array(step, columns[1]), np.asarray([1], dtype=np.int64))


def test_same_filename_and_schema_in_different_directories_have_unique_part_ids(tmp_path: Path):
    first = tmp_path / "trajs" / "suite" / "task_a" / "v2" / "robot_v2.pkl"
    second = tmp_path / "trajs" / "suite" / "task_b" / "v2" / "robot_v2.pkl"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"franka": [_episode()]}, handle)

    plan = inspect_collection(tmp_path)

    assert len(plan.parts) == 2
    assert len({part.part_id for part in plan.parts}) == 2
