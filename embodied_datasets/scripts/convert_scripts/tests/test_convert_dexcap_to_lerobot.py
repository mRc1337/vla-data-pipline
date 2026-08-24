from types import SimpleNamespace

def test_dexcap_estimate_uses_dexcap_benchmark_ratio() -> None:
    import convert_arcap_to_lerobot as pipeline

    saved = {
        key: getattr(pipeline, key)
        for key in (
            "DEFAULT_RAW_ROOT",
            "DEFAULT_LOCAL_WORK_ROOT",
            "PARTITION_SPECS",
            "OFFICIAL_PARTITIONS",
            "READER_FORMAT",
            "PIPELINE_LABEL",
            "SOURCE_DATASET",
            "COLLECTION_KIND",
            "inspect_partition",
            "iter_frames",
            "_conversion_options",
            "_estimate",
            "_manifest",
        )
    }
    try:
        from convert_dexcap_to_lerobot import _estimate

        info = SimpleNamespace(
            all_frame_count=100,
            plan=SimpleNamespace(num_frames=10),
            selected_logical_bytes=0,
            spec=SimpleNamespace(source_bytes=0),
        )

        estimate = _estimate([info], [])
    finally:
        for key, value in saved.items():
            setattr(pipeline, key, value)

    assert estimate["full_dataset_frames"] == 100
    assert estimate["expected_full_output_bytes"] < 1_000_000_000
    assert estimate["expected_output_bytes"] < estimate["expected_full_output_bytes"]
