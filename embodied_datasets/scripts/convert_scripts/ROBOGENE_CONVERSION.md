# RoboGene v2.1 to LeRobot v3

`convert_robogene_to_lerobot.py` converts RoboGene's task-local legacy
LeRobot v2.1 releases without decoding or concatenating the native RGB H.264
MP4 payloads.  It preserves all source Parquet payload fields and only
rewrites generated `episode_index`, `index`, and `task_index` columns.

The reader partitions by top-level robot split and Parquet schema fingerprint.
The normal release therefore writes `robogene/dual_arm` and
`robogene/single_arm`; a split containing multiple schemas receives stable
`--schema-<hash>` suffixes instead of mixing incompatible fields.

Runtime files are confined to `$HOME/robogene_staging`.  Each verified
local unit is copied to its final OSS chunk name, validated there (size,
sample ranges, Parquet footer/schema, and video container metadata), then its
local bulk files are removed.  Compact unit metadata and a frozen preflight
catalog remain locally for resume.  `--resume` validates recorded source
size/mtime without re-traversing raw directories and uses the original run ID.

The normal initial/background command deliberately includes `--resume`; on a
fresh output it initializes the resume state, and on an incomplete output it
requires that existing state.

```bash
cd $HOME/vla-data-pipeline
PYTHONPATH=embodied_datasets/scripts/convert_scripts .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_robogene_to_lerobot.py \
  --resume --workers 4 --encoder-threads-per-worker 8 --upload-workers 1 \
  --max-inflight-units 4 --max-local-temp-bytes 100000000000 \
  --min-local-free-bytes 200000000000
```

Useful bounded checks:

```bash
# Metadata-only exact task check; no state/output writes.
PYTHONPATH=embodied_datasets/scripts/convert_scripts .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_robogene_to_lerobot.py \
  --inspect-only --task robogene_twoArm_franka_adjust_lamp --limit-episodes 1

# Convert a small exact selection to the approved staging output.
PYTHONPATH=embodied_datasets/scripts/convert_scripts .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_robogene_to_lerobot.py \
  --resume --task robogene_twoArm_franka_adjust_lamp --limit-episodes 1
```

`--limit-tasks`, `--limit-episodes`, and `--limit-shards` are deterministic
lexical smoke/benchmark selections.  A source shard is a task-local
`data/chunk-*` directory.

The field mapping is intentionally conservative:

| Source | LeRobot v3 handling |
|---|---|
| `action.*`, `observation.state.*` | Same field, shape, dtype, names, and values |
| `observation.rgb_images.*` | Same RGB H.264 stream, copied to the canonical v3 video path |
| `observation.depth_images.*` | Same image struct payload and shape, preserved in Parquet |
| `timestamp`, `frame_index` | Copied without resampling or normalization |
| `episode_index`, `index`, `task_index` | Coordinator-assigned global values, written during the single local Parquet rewrite |
| `tasks.jsonl` / episode `tasks` | Episode instruction and final `tasks.parquet` mapping |
