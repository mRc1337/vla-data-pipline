from __future__ import annotations

from pathlib import Path

from convert_core.performance import ProcessTreeSampler


def test_process_tree_sampler_reports_work_and_peak_temp(tmp_path: Path):
    (tmp_path / "existing.bin").write_bytes(b"z" * 64_000)
    sampler = ProcessTreeSampler(tmp_path, interval_seconds=0.01)
    sampler.start()
    payload = b"x" * 128_000
    (tmp_path / "sample.bin").write_bytes(payload)
    total = 0
    for value in range(200_000):
        total += value * value
    metrics = sampler.stop()

    assert total > 0
    assert metrics.wall_seconds > 0
    assert metrics.cpu_seconds >= 0
    assert metrics.average_cpu_cores >= 0
    assert metrics.peak_rss_bytes > 0
    assert metrics.io_counters_available
    assert metrics.write_chars >= len(payload)
    assert metrics.peak_temp_bytes >= len(payload)
    assert metrics.peak_temp_bytes < len(payload) + 64_000
