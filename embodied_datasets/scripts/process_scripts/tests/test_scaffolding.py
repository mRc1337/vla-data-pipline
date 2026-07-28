from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

EXPECTED_DIRS = [
    "embodied_datasets/scripts/process_scripts/registry_configs",
    "embodied_datasets/scripts/process_scripts/configs",
    "embodied_datasets/data_root/public_datasets_raw",
    "embodied_datasets/data_root/public_datasets_staging/lerobot_v3_0",
    "embodied_datasets/data_root/public_datasets/lerobot_v3_0",
    "embodied_datasets/data_root/urdf_assets",
]


def test_expected_directories_exist():
    for rel_dir in EXPECTED_DIRS:
        assert (REPO_ROOT / rel_dir).is_dir(), f"missing directory: {rel_dir}"
