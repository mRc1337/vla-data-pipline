import csv
import io

from common.generate_full_export import render_full_csv
from common.schema import DatasetConfig


def test_render_full_csv_includes_all_schema_fields_as_header():
    csv_text = render_full_csv([])
    header_line = csv_text.strip().splitlines()[0]
    columns = next(csv.reader(io.StringIO(header_line)))
    assert columns == list(DatasetConfig.model_fields.keys())


def test_render_full_csv_one_row_per_dataset():
    configs = [
        DatasetConfig(id="a", name="A"),
        DatasetConfig(id="b", name="B"),
    ]
    csv_text = render_full_csv(configs)
    lines = csv_text.strip().splitlines()
    assert len(lines) == 3


def test_render_full_csv_formats_enum_and_list_and_dict_fields():
    config = DatasetConfig(
        id="droid",
        name="DROID",
        license="MIT",
        camera_views=["third_person", "wrist"],
        field_sources={"license": "https://example.com"},
    )
    csv_text = render_full_csv([config])
    row = next(csv.DictReader(io.StringIO(csv_text)))
    assert row["license"] == "MIT"
    assert row["camera_views"] == "third_person; wrist"
    assert "license=https://example.com" in row["field_sources"]


def test_render_full_csv_blank_for_none_and_empty_fields():
    config = DatasetConfig(id="droid", name="DROID")
    csv_text = render_full_csv([config])
    row = next(csv.DictReader(io.StringIO(csv_text)))
    assert row["source_url"] == ""
    assert row["camera_views"] == ""
    assert row["field_sources"] == ""
