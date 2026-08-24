import json
from pathlib import Path

import numpy as np

from convert_core.equivalence import _semantic_manifest, _video_frames_equal


def test_semantic_manifest_ignores_decoder_execution_options(tmp_path: Path) -> None:
    path = tmp_path / "conversion_manifest.json"
    path.write_text(
        json.dumps(
            {
                "decoder": {
                    "batch_size": 8,
                    "v1_postprocess_device": "gpu",
                    "decoder_cpu_threads": 8,
                },
                "episodes": [],
            }
        ),
        encoding="utf-8",
    )
    assert _semantic_manifest(path) == {"decoder": {"batch_size": 8}, "episodes": []}


def test_lossy_video_equivalence_allows_small_quantization_difference() -> None:
    reference = np.zeros((8, 8, 3), dtype=np.uint8)
    candidate = reference.copy()
    candidate[::2, :, 0] = 12

    assert _video_frames_equal(reference, candidate)


def test_lossy_video_equivalence_rejects_content_difference() -> None:
    reference = np.zeros((8, 8, 3), dtype=np.uint8)
    candidate = np.full_like(reference, 64)

    assert not _video_frames_equal(reference, candidate)
