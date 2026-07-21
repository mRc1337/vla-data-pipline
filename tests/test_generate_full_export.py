from common.generate_full_export import _COLUMNS, render_full_table_markdown
from common.schema import DatasetConfig, Priority, RegistryEntry


def _entry(id_, name="Name", priority=Priority.P2):
    return RegistryEntry(id=id_, name=name, priority=priority)


def test_render_full_table_markdown_header_excludes_provenance_columns():
    table = render_full_table_markdown([], {})
    header_line = table.splitlines()[0]
    assert "field_sources" not in header_line
    assert "suggested_new_enum_values" not in header_line
    assert "field_sources" not in _COLUMNS
    assert "suggested_new_enum_values" not in _COLUMNS
    for column in DatasetConfig.model_fields.keys():
        if column not in ("field_sources", "suggested_new_enum_values"):
            assert column in header_line


def test_render_full_table_markdown_one_row_per_dataset():
    entries = [_entry("a"), _entry("b")]
    configs_by_id = {
        "a": DatasetConfig(id="a", name="A"),
        "b": DatasetConfig(id="b", name="B"),
    }
    table = render_full_table_markdown(entries, configs_by_id)
    lines = table.strip().splitlines()
    assert len(lines) == 4  # header + separator + 2 rows


def test_render_full_table_markdown_formats_enum_and_list_fields():
    entries = [_entry("droid", name="DROID")]
    configs_by_id = {
        "droid": DatasetConfig(
            id="droid",
            name="DROID",
            license="MIT",
            camera_views=["third_person", "wrist"],
        )
    }
    table = render_full_table_markdown(entries, configs_by_id)
    row_line = table.strip().splitlines()[2]
    assert "MIT" in row_line
    assert "third_person; wrist" in row_line


def test_render_full_table_markdown_blank_for_none_fields():
    entries = [_entry("droid", name="DROID")]
    configs_by_id = {"droid": DatasetConfig(id="droid", name="DROID")}
    table = render_full_table_markdown(entries, configs_by_id)
    row_line = table.strip().splitlines()[2]
    cells = [c.strip() for c in row_line.strip("|").split("|")]
    source_url_index = _COLUMNS.index("source_url")
    assert cells[source_url_index] == ""


def test_render_full_table_markdown_sorts_by_priority_then_name():
    entries = [
        _entry("z_low", name="Z Low Priority", priority=Priority.P2),
        _entry("a_high", name="A High Priority", priority=Priority.P0),
    ]
    configs_by_id = {
        "z_low": DatasetConfig(id="z_low", name="Z Low Priority"),
        "a_high": DatasetConfig(id="a_high", name="A High Priority"),
    }
    table = render_full_table_markdown(entries, configs_by_id)
    data_lines = table.strip().splitlines()[2:]
    assert "a_high" in data_lines[0]
    assert "z_low" in data_lines[1]
