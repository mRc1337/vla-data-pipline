# RoboCOIN v2.1 → LeRobot v3.0

`convert_robocoin_to_lerobot.py` converts the 14 publisher task directories in
stable case-insensitive name order.  It never writes the source tree.  It
copies native MP4 containers, rewrites only generated indices and the Arrow
list container representation required by LeRobot v3, then commits one whole
task per deterministic chunk.

RoboCOIN contains seven mutually incompatible feature schemas (different
robots, state/action widths, and camera sets).  A single LeRobot dataset cannot
represent them without padding or dropping fields.  The output is therefore a
collection of seven standard LeRobot v3 datasets under `robocoin/schema-*`.
Canonical indices are contiguous inside each schema partition; the root
`conversion_manifest.json` additionally records deterministic collection-wide
episode/frame ranges.  Each partition can be opened directly with
`LeRobotDataset`; the collection root intentionally is not one homogeneous
`LeRobotDataset`.

## Field mapping

| 原字段 | shape/dtype | 语义/单位 | LeRobot 字段 | 转换 | 依据 | 是否有损 |
|---|---|---|---|---|---|---|
| `observation.state` | per-task `[26]`, `[28]`, `[30]`, `[34]`, `[36]`, `[49]`, or `[118]`; declared float32 | publisher-named joint, EEF, hand, IMU values; units are encoded in component names (`rad`, `m`, `rad_s`, `Nm`, etc.) | same | v2 variable list becomes v3 fixed-size list; the Agilex/aloha partition uses float64 because its Parquet payload mixes float32/float64, so float32 episodes are losslessly widened and float64 episodes remain float64; other partitions retain their declared dtype | task `meta/info.json`, README, every Parquet footer | 否 |
| `action` | per-task `[16]`, `[26]`, `[28]`, `[30]`, `[34]`, `[36]`, or `[54]`; declared float32 | publisher-named commanded joint/EEF/hand values with units in component names | same | same common-supertype rule as `observation.state`; no float64→float32 narrowing is permitted | same | 否 |
| `subtask_annotation`, `scene_annotation` | `[5]`, `[1]` / int32 where present | publisher annotation codes | same | copied; singleton list becomes v3 scalar convention | `meta/info.json`, Parquet footer | 否 |
| `eef_*`, `gripper_*` auxiliary fields | task-declared `[2]` or `[12]` / float32, float64, or int32 | publisher-derived EEF/gripper pose, direction, velocity, magnitude, mode/activity/scale | same | values/order copied; Agilex `gripper_open_scale_{state,action}` is physically float64 despite declared float32 and therefore remains float64 | `meta/info.json`, README, every Parquet footer | 否 |
| `observation.images.*` | task-declared H×W×3 / video | named RGB camera stream | same | MP4 byte copy; no decode, re-encode, resample, or resize | `meta/info.json`, README, MP4 headers | 否 |
| `timestamp` | scalar float32 or float64 | source timestamp at 30 FPS | same | copied without regeneration/resampling | Parquet footer and sampled values | 否 |
| `frame_index` | scalar int64 | episode-local zero-based frame ordinal | same | copied and checked contiguous | Parquet index column | 否 |
| `episode_index`, `index`, `task_index` | scalar int64 | generated partition-global episode/frame/task identifiers | same | deterministically rebased from catalog ranges and frame-level source task mapping | Parquet `task_index`, `tasks.jsonl`, task catalog | 否 |

Two Airbot paths, `cam_high_rgb` and `cam_third_view`, are not declared fields
in `info.json`, README, or `info.yaml`.  They are redundant publisher aliases
of `cam_head_rgb` and `cam_front_rgb`.  Preflight requires equal size and equal
first/middle/last 64 KiB digests for every episode before it omits those
duplicate objects; any mismatch is fatal.  Several task directories also have
empty or punctuation-drifted `episodes.jsonl.tasks`; the converter records the
discrepancy and uses the frame-level `task_index` → `tasks.jsonl` relationship
used by the source LeRobot loader.

Schema compatibility is conversion-policy version 2.  It fingerprints ordered
Arrow fields, recursive types, nullability, and field metadata, while excluding
top-level Pandas writer metadata such as the per-episode (and occasionally
stale) RangeIndex `stop`.  The two Agilex/aloha tasks contain both float32 and
float64 physical lists for `action` and `observation.state`, while both
`gripper_open_scale_*` fields are physically float64 throughout despite being
declared float32.  Their common output schema is explicitly float64 for all
four fields.  The policy version, all observed physical
schema variants, and the lossless dtype resolution are frozen into task
fingerprints and written into task and collection manifests.

## Resume and space behavior

The first launch writes a compact catalog using only `info.json`,
`tasks.jsonl`, and `episodes.jsonl`.  Each task is fully preflighted only when
it reaches the head of the queue, and its immutable plan freezes all source
relative paths, sizes/mtimes, schema fingerprints, mappings, media settings,
index ranges, and output UID.  A committed task is skipped from its marker
without source or remote-bulk reads.  An unfinished task re-stats only files in
its cached plan.

During local conversion, each completed episode gets an atomic
`.episode_checkpoints/episode-XXXXXX.json` marker containing the immutable task
fingerprint plus validated Parquet row/schema/index evidence and video
size/codec/resolution/FPS/frame-count evidence.  If conversion stops, valid
episode outputs are retained and checked in plan order; only a missing,
changed, or failed episode is discarded and rebuilt.  Task metadata and the
task-level verified marker are regenerated only after every episode checkpoint
is valid, so no partial task can be uploaded or committed.

There is no fixed total local-byte quota and no `--max-local-temp-bytes` option.
Before a task starts, its complete local peak is compared with filesystem
availability above `--min-local-free-bytes`; the streaming copy guard repeats
that check while writing.  Upload is single-threaded.  The direct commit order
is copy → size/range/footer/media verification → marker → local bulk deletion.
An upload failure retains the complete verified local task, and resume uploads
it without reconversion.  `_SUCCESS` is written only after all partitions,
metadata, stats, and the collection manifest finalize.

## Commands

Run the two-task preflight (no writes):

```bash
cd /home/pai/zxw/vla-data-pipeline
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=embodied_datasets/scripts/convert_scripts \
  .venv/bin/python embodied_datasets/scripts/convert_scripts/convert_robocoin_to_lerobot.py \
  --inspect-only \
  --task G1edu-u3_place_plastic_bowl_ag \
  --task Airbot_MMK2_storage_tools
```

Run the isolated two-task, one-episode smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=embodied_datasets/scripts/convert_scripts \
  .venv/bin/python embodied_datasets/scripts/convert_scripts/convert_robocoin_to_lerobot.py \
  --resume --output-dataset-uid robocoin_smoke \
  --task G1edu-u3_place_plastic_bowl_ag \
  --task Airbot_MMK2_storage_tools \
  --limit-episodes-per-task 1 \
  --workers 1 --encoder-threads-per-worker 8 --upload-workers 1 \
  --min-local-free-bytes 200000000000
```

Formal background conversion (provided only; not run):

```bash
mkdir -p /home/pai/zxw/robocoin_staging/logs
cd /home/pai/zxw/vla-data-pipeline
nohup env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=embodied_datasets/scripts/convert_scripts \
  .venv/bin/python embodied_datasets/scripts/convert_scripts/convert_robocoin_to_lerobot.py \
  --resume --workers 1 --encoder-threads-per-worker 8 --upload-workers 1 \
  --min-local-free-bytes 200000000000 \
  > /home/pai/zxw/robocoin_staging/logs/formal.log 2>&1 &
```

Status and resume:

```bash
find /home/pai/zxw/robocoin_staging/resume/committed -type f | sort
test -f /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/robocoin/_SUCCESS

# Resume with exactly the same semantic selection/output options.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=embodied_datasets/scripts/convert_scripts \
  .venv/bin/python embodied_datasets/scripts/convert_scripts/convert_robocoin_to_lerobot.py \
  --resume --workers 1 --encoder-threads-per-worker 8 --upload-workers 1 \
  --min-local-free-bytes 200000000000
```
