import openpyxl
import pytest

from common.migrate_xlsx_to_registry import (
    build_registry_and_configs,
    read_name_and_link_rows,
    slugify,
)


@pytest.fixture
def sample_xlsx(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "VLA公开数据集"
    ws.append(["序号", "名称", "数据类型", "官方下载链接"])
    ws.append([1, "DROID", None, "https://droid-dataset.github.io/"])
    ws.append([2, "BridgeData V2", None, None])
    ws.append([3, "Open X-Embodiment", None, "https://example.com/oxe"])
    path = tmp_path / "sample.xlsx"
    wb.save(path)
    return path


def test_slugify_basic():
    assert slugify("DROID") == "droid"
    assert slugify("BridgeData V2") == "bridgedata_v2"
    assert slugify("Open X-Embodiment") == "open_x_embodiment"
    assert slugify("lerobot/ull_folding") == "lerobot_ull_folding"


def test_slugify_rejects_empty():
    with pytest.raises(ValueError):
        slugify("   ")


def test_read_name_and_link_rows(sample_xlsx):
    rows = read_name_and_link_rows(sample_xlsx)
    assert rows == [
        ("DROID", "https://droid-dataset.github.io/"),
        ("BridgeData V2", None),
        ("Open X-Embodiment", "https://example.com/oxe"),
    ]


def test_build_registry_and_configs(sample_xlsx):
    rows = read_name_and_link_rows(sample_xlsx)
    entries, configs = build_registry_and_configs(rows)
    assert [e.id for e in entries] == ["droid", "bridgedata_v2", "open_x_embodiment"]
    assert entries[0].name == "DROID"
    assert entries[0].download_status.value == "not_downloaded"
    assert configs[0].source_url == "https://droid-dataset.github.io/"
    assert configs[1].source_url is None
    assert configs[0].review_status.value == "pending_human_review"


def test_build_registry_and_configs_rejects_duplicate_ids():
    rows = [("DROID", "https://a"), ("droid", "https://b")]
    with pytest.raises(ValueError):
        build_registry_and_configs(rows)
