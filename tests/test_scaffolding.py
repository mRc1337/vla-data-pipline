from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "embodied_datasets/public_datasets_raw/convert_scripts/configs",
    "embodied_datasets/public_datasets_raw/process_scripts/configs",
    "embodied_datasets/public_datasets_raw/verify_scripts/logs",
    "embodied_datasets/urdf_assets",
    "embodied_datasets/public_datasets/lerobot_v3_0",
]


def test_expected_directories_exist():
    for rel_dir in EXPECTED_DIRS:
        assert (REPO_ROOT / rel_dir).is_dir(), f"missing directory: {rel_dir}"
