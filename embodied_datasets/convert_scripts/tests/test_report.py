from common_convert.report import ConversionReport


def test_conversion_report_defaults():
    report = ConversionReport(num_episodes=5, num_frames=100)
    assert report.warnings == []
    assert report.urdf_path is None


def test_conversion_report_carries_warnings_and_urdf_path():
    report = ConversionReport(
        num_episodes=5,
        num_frames=100,
        warnings=["missing task field, defaulted to ''"],
        urdf_path="/data/urdf_assets/franka_panda/franka_panda.urdf",
    )
    assert report.warnings == ["missing task field, defaulted to ''"]
    assert report.urdf_path == "/data/urdf_assets/franka_panda/franka_panda.urdf"
