"""Per-raw_format download-integrity checkers, dispatched by
check_format(). See
docs/superpowers/specs/2026-07-21-convert-scripts-verify-scripts-design.md
section 6.

Deliberately has zero import-time dependency on convert_scripts/common
(the RawFormat enum lives there) -- check_format() compares raw_format
against plain string values (via `getattr(x, "value", x)`, the same
duck-typed pattern process_scripts/run_pipeline.py already uses for enum
fields) instead of importing the enum class. This sidesteps the
"convert_scripts/common and verify_scripts/common are both literally
named `common`" collision entirely for this module -- only run_verify.py
(which genuinely needs both) has to deal with it, via the same
importlib.util aliasing process_scripts/run_pipeline.py already uses.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional


class CheckOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NO_CHECKER = "no_checker"


@dataclass
class CheckResult:
    outcome: CheckOutcome
    reason: Optional[str] = None
    episode_count: Optional[int] = None

    @property
    def passed(self) -> bool:
        return self.outcome == CheckOutcome.PASSED


def check_hdf5(raw_path: Path, required_key_substrings: tuple = ("action", "obs")) -> CheckResult:
    """"能不能打开、必需字段存在"-- required_key_substrings 用递归 h5py.File.visit()
    在整个文件的键路径里找子串匹配（不假设固定的组层级），因为不同 HDF5
    数据集的分组深度差异很大（如 robomimic 的 data/demo_N/obs/...）。
    """
    import h5py

    hdf5_files = sorted(raw_path.rglob("*.hdf5")) + sorted(raw_path.rglob("*.h5"))
    if not hdf5_files:
        return CheckResult(outcome=CheckOutcome.FAILED, reason=f"no *.hdf5/*.h5 files found under {raw_path}")

    episode_count = 0
    for path in hdf5_files:
        try:
            with h5py.File(path, "r") as f:
                found = {substring: False for substring in required_key_substrings}

                def _visitor(name: str) -> None:
                    for substring in required_key_substrings:
                        if substring.lower() in name.lower():
                            found[substring] = True

                f.visit(_visitor)
                missing = [substring for substring, was_found in found.items() if not was_found]
                if missing:
                    return CheckResult(
                        outcome=CheckOutcome.FAILED, reason=f"{path.name}: missing required key substring(s) {missing}"
                    )
                if "data" in f:
                    episode_count += len(f["data"].keys())
        except OSError as exc:
            return CheckResult(outcome=CheckOutcome.FAILED, reason=f"{path.name}: cannot open as HDF5 ({exc})")

    return CheckResult(outcome=CheckOutcome.PASSED, episode_count=episode_count or None)


def check_rlds_or_tfrecord(raw_path: Path) -> CheckResult:
    """"完整遍历不抛异常"（design doc 第6节）-- 只用 tf.data.TFRecordDataset
    原始遍历，不用 tensorflow_datasets 解码具体字段结构（完整性校验不需要
    知道 RLDS 的 step/episode schema，只需要确认每条 record 能被读出来，
    没有 CRC 校验失败/截断等底层损坏）。
    """
    import tensorflow as tf

    tfrecord_files = sorted(str(p) for p in raw_path.rglob("*.tfrecord*"))
    if not tfrecord_files:
        return CheckResult(outcome=CheckOutcome.FAILED, reason=f"no *.tfrecord* files found under {raw_path}")

    try:
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        count = sum(1 for _ in dataset)
    except Exception as exc:  # tf raises several different exception types on a corrupt/truncated record
        return CheckResult(outcome=CheckOutcome.FAILED, reason=f"TFRecordDataset iteration failed: {exc}")

    if count == 0:
        return CheckResult(outcome=CheckOutcome.FAILED, reason="TFRecordDataset contained zero records")
    return CheckResult(outcome=CheckOutcome.PASSED, episode_count=count)


def check_lerobot(raw_path: Path) -> CheckResult:
    from shared.lerobot_io import load_lerobot_episodes

    try:
        episodes = load_lerobot_episodes(raw_path)
    except Exception as exc:  # lerobot's LeRobotDataset raises many different exception types on a malformed dataset
        return CheckResult(outcome=CheckOutcome.FAILED, reason=f"load_lerobot_episodes failed: {exc}")
    if not episodes:
        return CheckResult(outcome=CheckOutcome.FAILED, reason="load_lerobot_episodes returned zero episodes")
    return CheckResult(outcome=CheckOutcome.PASSED, episode_count=len(episodes))


CUSTOM_CHECKERS: Dict[str, Callable[[Path], CheckResult]] = {}


def check_custom(raw_path: Path, dataset_id: str) -> CheckResult:
    """No generic checker exists for Custom raw formats -- per-dataset
    checkers, when needed, get registered into CUSTOM_CHECKERS (keyed by
    dataset_id) rather than added to this function. Design doc section 6:
    a dataset without one must be recorded as skipped_no_checker, not
    silently skipped.
    """
    checker = CUSTOM_CHECKERS.get(dataset_id)
    if checker is None:
        return CheckResult(outcome=CheckOutcome.NO_CHECKER, reason=f"no per-dataset checker registered for {dataset_id!r}")
    return checker(raw_path)


def check_ros_bag_or_mcap(raw_path: Path, raw_format_value: str) -> CheckResult:
    """`rosbag info`/`mcap info`（design doc 第6节）验证头信息完整——两个都是
    外部 CLI 工具，不假设本机已装；缺失时返回 NO_CHECKER 而不是抛异常/静默
    跳过，跟 check_custom 未注册专属校验器时是同一个语义。
    """
    tool = "rosbag" if raw_format_value == "ROS_bag" else "mcap"
    if shutil.which(tool) is None:
        return CheckResult(outcome=CheckOutcome.NO_CHECKER, reason=f"{tool!r} CLI not installed on this machine")

    extension = ".bag" if raw_format_value == "ROS_bag" else ".mcap"
    files = sorted(raw_path.rglob(f"*{extension}"))
    if not files:
        return CheckResult(outcome=CheckOutcome.FAILED, reason=f"no *{extension} files found under {raw_path}")

    for path in files:
        result = subprocess.run([tool, "info", str(path)], capture_output=True, text=True)
        if result.returncode != 0:
            return CheckResult(outcome=CheckOutcome.FAILED, reason=f"{tool} info failed on {path.name}: {result.stderr.strip()[:200]}")
    return CheckResult(outcome=CheckOutcome.PASSED)


def check_scale(
    actual_size_gb: float,
    actual_episode_count: Optional[int],
    expected_size_gb: Optional[float],
    expected_num_episodes: Optional[int],
    tolerance: float = 0.5,
) -> CheckResult:
    """Flags "most of the data is missing" -- coarse on purpose
    (+/-50% by default), not a precise reconciliation. Either dimension
    is skipped when its onboarding-declared expected value, or (for
    episode count) the actual count itself, isn't available.
    """
    reasons = []
    if expected_size_gb:
        ratio = actual_size_gb / expected_size_gb
        if abs(ratio - 1.0) > tolerance:
            reasons.append(f"actual size {actual_size_gb:.2f}GB vs expected {expected_size_gb:.2f}GB (ratio {ratio:.2f})")
    if expected_num_episodes and actual_episode_count is not None:
        ratio = actual_episode_count / expected_num_episodes
        if abs(ratio - 1.0) > tolerance:
            reasons.append(
                f"actual {actual_episode_count} episodes vs expected {expected_num_episodes} (ratio {ratio:.2f})"
            )
    if reasons:
        return CheckResult(outcome=CheckOutcome.FAILED, reason="; ".join(reasons))
    return CheckResult(outcome=CheckOutcome.PASSED)


_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def find_video_files(raw_path: Path) -> List[Path]:
    return sorted(p for p in raw_path.rglob("*") if p.suffix.lower() in _VIDEO_EXTENSIONS)


def check_video_decodable(video_paths: List[Path]) -> CheckResult:
    """Uses `ffprobe` (already a documented system dependency for
    process_scripts' real-video tests, not a new one) rather than the
    `decord` python package design doc section 5 also lists -- ffprobe
    already gives a sufficient "can this be decoded" signal for a coarse
    integrity check, and adding decord as a new pip dependency for the
    same signal wasn't judged worth it (see plan Self-Review).
    """
    if shutil.which("ffprobe") is None:
        return CheckResult(outcome=CheckOutcome.NO_CHECKER, reason="ffprobe not installed on this machine")
    for path in video_paths:
        result = subprocess.run(["ffprobe", "-v", "error", str(path)], capture_output=True, text=True)
        if result.returncode != 0 or result.stderr.strip():
            return CheckResult(
                outcome=CheckOutcome.FAILED, reason=f"ffprobe reported an error on {path.name}: {result.stderr.strip()[:200]}"
            )
    return CheckResult(outcome=CheckOutcome.PASSED)


def check_format(raw_path: Path, raw_format, dataset_id: str) -> CheckResult:
    raw_format_value = getattr(raw_format, "value", raw_format)
    if raw_format_value in ("RLDS", "TFRecord"):
        return check_rlds_or_tfrecord(raw_path)
    if raw_format_value in ("ROS_bag", "MCAP"):
        return check_ros_bag_or_mcap(raw_path, raw_format_value)
    if raw_format_value == "HDF5":
        return check_hdf5(raw_path)
    if raw_format_value == "LeRobot":
        return check_lerobot(raw_path)
    if raw_format_value == "Custom":
        return check_custom(raw_path, dataset_id)
    return CheckResult(
        outcome=CheckOutcome.NO_CHECKER, reason=f"no checker implemented for raw_format {raw_format_value!r}"
    )
