import numpy as np

from views import action_bands, band_overlay_figure, episode_status_label, slice_band, state_bands


def test_episode_status_label_kept():
    record = {"survived": True, "stages": [{"stage": "stage1_sudden_change", "rejected": False}]}
    assert episode_status_label(record) == "kept"


def test_episode_status_label_rejected_at_first_rejecting_stage():
    record = {
        "survived": False,
        "stages": [
            {"stage": "stage1_sudden_change", "rejected": False},
            {"stage": "stage2_trend_alignment", "rejected": True},
        ],
    }
    assert episode_status_label(record) == "rejected at stage2_trend_alignment"


def test_episode_status_label_dropped_all_frames():
    record = {"survived": False, "stages": [{"stage": "stage1_sudden_change", "rejected": False}]}
    assert episode_status_label(record) == "dropped (all frames removed)"


def test_state_bands_single_arm_matches_128dim_layout():
    bands = state_bands(num_arms=1)
    labels = [(b.label, b.start, b.end) for b in bands]
    assert labels == [
        ("arm1_joint", 0, 7),
        ("arm1_eef", 7, 14),
        ("arm1_gripper", 14, 35),
        ("reserve", 35, 128),
    ]


def test_state_bands_dual_arm_matches_128dim_layout():
    bands = state_bands(num_arms=2)
    labels = [(b.label, b.start, b.end) for b in bands]
    assert labels == [
        ("arm1_joint", 0, 7),
        ("arm1_eef", 7, 14),
        ("arm1_gripper", 14, 35),
        ("arm2_joint", 35, 42),
        ("arm2_eef", 42, 49),
        ("arm2_gripper", 49, 70),
        ("reserve", 70, 128),
    ]


def test_action_bands_single_arm_matches_128dim_layout():
    bands = action_bands(num_arms=1)
    labels = [(b.label, b.start, b.end) for b in bands]
    assert labels == [
        ("arm1_joint", 0, 7),
        ("arm1_eef_pos", 7, 10),
        ("arm1_eef_rot", 10, 13),
        ("arm1_gripper", 14, 35),
        ("reserve", 35, 128),
    ]


def test_action_bands_dual_arm_matches_128dim_layout():
    bands = action_bands(num_arms=2)
    labels = [(b.label, b.start, b.end) for b in bands]
    assert labels == [
        ("arm1_joint", 0, 7),
        ("arm1_eef_pos", 7, 10),
        ("arm1_eef_rot", 10, 13),
        ("arm1_gripper", 14, 35),
        ("arm2_joint", 35, 42),
        ("arm2_eef_pos", 42, 45),
        ("arm2_eef_rot", 45, 48),
        ("arm2_gripper", 49, 70),
        ("reserve", 70, 128),
    ]


def test_slice_band_reports_unpopulated_band():
    vector = np.zeros((3, 128))
    mask = np.zeros(128, dtype=bool)
    mask[0:7] = True  # only arm1_joint populated
    band = state_bands(num_arms=1)[1]  # arm1_eef
    result = slice_band(vector, mask, band)
    assert result["populated"] is False
    assert result["values"].shape == (3, 7)


def test_slice_band_reports_populated_band():
    vector = np.zeros((3, 128))
    mask = np.zeros(128, dtype=bool)
    mask[0:7] = True
    band = state_bands(num_arms=1)[0]  # arm1_joint
    result = slice_band(vector, mask, band)
    assert result["populated"] is True


def test_slice_band_without_mask_defaults_to_populated():
    vector = np.zeros((3, 128))
    band = state_bands(num_arms=1)[0]
    result = slice_band(vector, None, band)
    assert result["populated"] is True
    assert result["mask"] is None


def test_band_overlay_figure_raw_only_has_one_trace_per_dim():
    band = state_bands(num_arms=1)[0]  # arm1_joint, dim 7
    raw_slice = slice_band(np.zeros((5, 128)), None, band)
    fig = band_overlay_figure(band.label, raw_slice, None)
    assert len(fig.data) == 7
    assert all(trace.name.endswith("(raw)") for trace in fig.data)


def test_band_overlay_figure_with_final_doubles_trace_count_and_shares_color():
    band = state_bands(num_arms=1)[0]  # arm1_joint, dim 7
    raw_slice = slice_band(np.zeros((5, 128)), None, band)
    mask = np.ones(128, dtype=bool)
    final_slice = slice_band(np.zeros((3, 128)), mask, band)
    fig = band_overlay_figure(band.label, raw_slice, final_slice)
    assert len(fig.data) == 14
    raw_traces = [t for t in fig.data if t.name.endswith("(raw)")]
    final_traces = [t for t in fig.data if t.name.endswith("(final)")]
    assert len(raw_traces) == len(final_traces) == 7
    # dim0's raw and final traces should share a color for the overlay to read.
    assert raw_traces[0].line.color == final_traces[0].line.color
    assert raw_traces[0].line.dash == "dot"
    assert final_traces[0].line.dash is None
