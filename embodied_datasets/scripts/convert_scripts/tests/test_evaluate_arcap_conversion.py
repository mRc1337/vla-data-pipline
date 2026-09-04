from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from evaluate_arcap_conversion import _find_rows


def test_find_rows_batches_selected_indices_across_row_groups(tmp_path: Path) -> None:
    data = tmp_path / "data" / "chunk-000"
    data.mkdir(parents=True)
    pq.write_table(
        pa.table({"index": list(range(6)), "value": [value * 10 for value in range(6)]}),
        data / "file-000.parquet",
        row_group_size=2,
    )

    rows = _find_rows(tmp_path, [1, 4], ["index", "value"])

    assert rows == {
        1: {"index": 1, "value": 10},
        4: {"index": 4, "value": 40},
    }
