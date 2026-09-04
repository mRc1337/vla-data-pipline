from pathlib import Path

import numpy as np
import pytest

from convert_core.dataset_config import CameraFieldConfig, DatasetConversionConfig, VectorFieldConfig
from convert_core.errors import ConversionError
from readers.rlds_reader import (
    RldsReader,
    _camera_shape,
    _get_by_path,
    _materialize_steps,
    _resolve_instruction,
    _validate_vector_step,
)

# tensorflow_datasets is not installed even in this project's own conda env
# as of this writing (see PIPELINE_STATUS.md), so these tests exercise every
# piece of readers/rlds_reader.py's logic that does not require the library
# itself -- path lookup, instruction resolution, vector/camera validation --
# using plain dicts/numpy arrays in place of a real TFDS episode. The one
# thing genuinely untested here is RldsReader.build_plan's actual
# tfds.builder_from_directory/as_dataset calls; see the "without tfds
# installed" test below and the module docstring for what that leaves open.


def test_get_by_path_success_and_missing():
    step = {"observation": {"state": [1, 2, 3]}, "action": [4, 5]}
    assert _get_by_path(step, "observation/state") == [1, 2, 3]
    assert _get_by_path(step, "action") == [4, 5]
    with pytest.raises(KeyError):
        _get_by_path(step, "observation/missing")
    with pytest.raises(KeyError):
        _get_by_path(step, "action/0")


def test_materialize_steps_accepts_a_plain_list_without_needing_tfds():
    steps = [{"a": 1}, {"a": 2}]
    assert _materialize_steps(steps, tfds=None) == steps


def _config(**overrides) -> DatasetConversionConfig:
    defaults: dict = dict(
        dataset_uid="rlds_test",
        format="rlds",
        robot_type="test_robot",
        fps=10,
        vector_fields=[VectorFieldConfig(feature_key="observation.state", source_key="observation/state", dim=3)],
        cameras=[CameraFieldConfig(feature_key="observation.images.primary", source_key="observation/image")],
    )
    defaults.update(overrides)
    return DatasetConversionConfig(**defaults)


def test_resolve_instruction_constant():
    config = _config(instruction_source="constant", instruction_constant="do it")
    assert _resolve_instruction(config, {}) == "do it"


def test_resolve_instruction_constant_requires_value_to_be_set():
    config = _config(instruction_source="constant", instruction_constant=None)
    with pytest.raises(ConversionError, match="instruction_constant"):
        _resolve_instruction(config, {})


def test_resolve_instruction_field_decodes_bytes():
    config = _config(instruction_source="field", instruction_field="language_instruction")
    step = {"language_instruction": b"pick up the cup"}
    assert _resolve_instruction(config, step) == "pick up the cup"


def test_resolve_instruction_field_defaults_to_language_instruction():
    config = _config(instruction_source="field")
    step = {"language_instruction": "pick up the cup"}
    assert _resolve_instruction(config, step) == "pick up the cup"


def test_resolve_instruction_field_missing_raises():
    config = _config(instruction_source="field", instruction_field="language_instruction")
    with pytest.raises(ConversionError, match="missing instruction field"):
        _resolve_instruction(config, {})


def test_resolve_instruction_path_parent_not_supported_for_rlds():
    config = _config(instruction_source="path_parent")
    with pytest.raises(ConversionError, match="path_parent"):
        _resolve_instruction(config, {})


def test_validate_vector_step_success_and_size_mismatch():
    field = VectorFieldConfig(feature_key="observation.state", source_key="observation/state", dim=3)
    _validate_vector_step({"observation": {"state": [1.0, 2.0, 3.0]}}, field, episode_index=0)
    with pytest.raises(ConversionError, match="expected dim=3"):
        _validate_vector_step({"observation": {"state": [1.0, 2.0]}}, field, episode_index=0)


def test_validate_vector_step_missing_field_raises():
    field = VectorFieldConfig(feature_key="observation.state", source_key="observation/state", dim=3)
    with pytest.raises(ConversionError, match="missing field"):
        _validate_vector_step({"observation": {}}, field, episode_index=0)


def test_camera_shape_success_and_wrong_dtype_raises():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    assert _camera_shape({"observation": {"image": image}}, "observation/image", "cam", episode_index=0) == (4, 5)

    float_image = image.astype(np.float32)
    with pytest.raises(ConversionError, match="not a decoded"):
        _camera_shape({"observation": {"image": float_image}}, "observation/image", "cam", episode_index=0)


def test_build_plan_raises_clear_error_when_tensorflow_datasets_is_missing(tmp_path: Path):
    try:
        import tensorflow_datasets  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("tensorflow_datasets is installed; this test only covers its absence")

    with pytest.raises(RuntimeError, match="tensorflow_datasets is required"):
        RldsReader().build_plan(_config(), tmp_path / "raw", tmp_path / "staging")
