from common.generate_overview_readme import (
    TABLE_END_MARKER,
    TABLE_START_MARKER,
    render_overview_table,
    replace_marked_section,
)
from common.schema import DatasetConfig, RegistryEntry


def test_render_overview_table_sorts_by_priority_then_name():
    entries = [
        RegistryEntry(id="b", name="Bravo", priority="P1"),
        RegistryEntry(id="a", name="Alpha", priority="P0"),
        RegistryEntry(id="c", name="Charlie", priority="P0"),
    ]
    table = render_overview_table(entries, {})
    data_lines = table.splitlines()[2:]
    ids_in_order = [line.split("|")[1].strip() for line in data_lines]
    assert ids_in_order == ["a", "c", "b"]


def test_render_overview_table_includes_config_fields():
    entries = [RegistryEntry(id="droid", name="DROID")]
    configs = {
        "droid": DatasetConfig(
            id="droid",
            name="DROID",
            collection_method="teleop",
            embodiment_class="single_arm",
        )
    }
    table = render_overview_table(entries, configs)
    assert "teleop" in table
    assert "single_arm" in table


def test_render_overview_table_handles_missing_config():
    entries = [RegistryEntry(id="droid", name="DROID")]
    table = render_overview_table(entries, {})
    assert "droid" in table


def test_replace_marked_section_only_touches_between_markers():
    readme_text = (
        f"# Title\n\nIntro text.\n\n"
        f"{TABLE_START_MARKER}\nold table\n{TABLE_END_MARKER}\n\n"
        f"Footer text.\n"
    )
    updated = replace_marked_section(readme_text, "new table")
    assert "Intro text." in updated
    assert "Footer text." in updated
    assert "old table" not in updated
    assert "new table" in updated
