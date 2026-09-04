# DexCap → LeRobot v3.0 staging

This converter covers only
`/mnt/data/embodied_datasets/public_datasets_raw/dexcap` →
`/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/dexcap`.
It does not clean, normalize, resample, map to a 128-D embodiment schema, or
write to `public_datasets`.

## Source inventory and semantics

The released HDF5 files contain no timestamp/FPS field. The validated release
convention is therefore represented as 10 Hz with timestamps
`frame_index / 10`. Metadata inspection checks every episode's required schema,
dtype, frame count, and time alignment; payload inspection samples first,
middle, and last frames and checks the stored action relationship.

| Source field | Shape / dtype | LeRobot field | Conversion | Lossy |
|---|---|---|---|---|
| `actions` | `[T,46]` float64 | `action` | identity; stored `actions[t]` is the next-frame EEF target | no |
| `glove_states` | `[T,63]` float64 | `observation.glove_state` | identity | no |
| `obs/robot0_eef_hand` | `[T,32]` float64 | `observation.eef_hand` | identity | no |
| `obs/robot0_eef_pos` | `[T,6]` float64 | `observation.eef_position` | identity | no |
| `obs/robot0_eef_quat` | `[T,8]` float64 | `observation.eef_quaternion` | identity | no |
| `obs/pointcloud` | `[T,10000,6]` float64 | `observation.pointcloud` | identity XYZRGB mapping | no |
| `states` | `[T,16]` float64 | `observation.source_state` | identity | no |
| `rewards` | `[T]` float64 | `source.reward` | scalar represented as a one-element vector at ingestion | no |
| `dones` | `[T]` int64 | `source.done` | scalar represented as a one-element vector at ingestion | no |
| `obs/label` | `[T]` int64 | `source.label` | scalar represented as a one-element vector at ingestion | no |
| `obs/agentview_image` | `[T,84,84,3]` uint8 | `observation.images.agentview` | streamed CPU H.264, CRF 18 | yes |

The two deterministic partitions are:

| Partition | Episodes | Frames | Source bytes |
|---|---:|---:|---:|
| `packaging_wild` | 480 | 57,916 | 29,108,980,760 |
| `wiping` | 255 | 27,901 | 14,023,370,344 |

The full release contains 735 episodes and 85,817 frames. No source field is
cropped, padded, reordered, cast, or silently dropped.

## Bounded publication design

The coordinator freezes the plan and global episode/frame/task ranges before
dispatch. Each worker writes one complete unit below
`$HOME/dexcap_staging/work/<run_id>/unit-XXXXXX/`. The unit is reopened
and validated before its Parquet and MP4 files are copied directly to their
final chunk paths under the OSS staging output. The uploader validates size,
Parquet footer/schema or MP4 codec/shape/FPS/frame count, and first/middle/last
byte samples. Only then are local bulk files deleted; compact metadata,
fingerprints, and verified markers remain under local `resume/`.

The local ledger accounts `work`, upload queues, `resume`, `logs`, and `cache`
together. It reserves estimated unit peaks before dispatch, blocks when the
100,000,000,000-byte ceiling would be exceeded, and requires at least
200,000,000,000 bytes of filesystem availability. Failed uploads retain their
local source and remove invalid remote destinations. `--resume` revalidates
remote evidence and retransmits only missing or corrupt units. `_SUCCESS` is
the publication gate; valid existing output is never overwritten.

The real bounded benchmark used 8 `packaging_wild` episodes / 1,108 frames and
all 1/2/4 conversion-worker × 1/2 uploader-worker combinations. All six
candidates were semantically equivalent with zero failures/retries; the
selected configuration was 4 conversion workers × 2 upload workers:
23.753 frames/s, 1.407× W1/U1 speedup, about 13.5 GB peak process-tree RSS,
and 577,954,134 bytes peak temporary usage. Independent H.264 processes can
produce small lossy quantization differences, so decoded video comparison uses
a documented bounded pixel tolerance; Parquet/schema/index/task/metadata
comparison remains exact.

The benchmark-derived full-release output estimate is approximately
28,346,857,053 bytes, with a 15% conservative estimate of 32,598,885,611
bytes. This is a planning estimate only; no full conversion was started.

## Commands

Read-only preflight:

```bash
.venv/bin/python embodied_datasets/scripts/convert_scripts/convert_dexcap_to_lerobot.py \
  --inspect-only --estimate-storage
```

Formal command (not executed during this task):

```bash
.venv/bin/python embodied_datasets/scripts/convert_scripts/convert_dexcap_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/dexcap \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --local-work-root $HOME/dexcap_staging \
  --output-dataset-uid dexcap \
  --acceleration-mode parallel \
  --workers 4 --encoder-threads-per-worker 8 \
  --upload-workers 2 --max-inflight-units 4 \
  --max-local-temp-bytes 100000000000 \
  --min-local-free-bytes 200000000000
```

Recovery uses the same command with `--resume` added. Smoke, benchmark, and
source-inventory checks completed without leaving output or local runtime
artifacts; the raw source files remain unchanged.
