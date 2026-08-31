# Functional Manipulation Benchmark conversion

The production converter consumes only the completed extracted release:

```text
/mnt/data/embodied_datasets/public_datasets_raw/functional_manipulation_benchmark_fmb_extracted/
├── assembly_1/
├── assembly_2/
├── assembly_3/
└── single_object/
```

Each shard must contain `.EXTRACTED_OK`. ZIP files are intentionally rejected
by the production entry point; the archive names remain only as logical
provenance and checkpoint labels.

`convert_fmb_to_lerobot.py` converts the public FMB release
`functional_manipulation_benchmark_fmb` to LeRobot v3.0.  The extracted source
contains compressed `.npy` object arrays; it is not RLDS.

## Source findings and mapping

The official [dataset card](https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark)
and its [dataset README](https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/raw/main/README.md)
document the fields and semantic ordering below, and the lazy NPY preflight
validates their shapes and dtypes for every trajectory without allocating the
array payload.  The [paper](https://arxiv.org/abs/2401.08553) documents the
SpaceMouse command rate as 10 Hz.  Physical units are not declared by the card
for the pose, velocity, force, torque, joint, Jacobian, or depth arrays; the
converter therefore preserves the raw values and does not invent units.
FMB stores camera arrays in BGR order although the card describes them as RGB,
so conversion reverses only the channel order before video encoding.

The official [TFDS builder](https://github.com/rail-berkeley/fmb/blob/main/fmb_dataset_builder/fmb_dataset/fmb_dataset_dataset_builder.py)
and [training dataloader](https://github.com/rail-berkeley/fmb/blob/main/ResNet/src/dataloader.py)
read the raw action array as `actions`, preserve `object_info` for
single-object trajectories, and use the same state/camera keys.  The reader
canonicalizes that observed plural export to the README's logical `action`
field; archives containing both spellings are rejected as ambiguous.

| Source field | Shape / dtype | Meaning | LeRobot field | Conversion | Evidence | Lossy |
|---|---|---|---|---|---|---|
| `obs/{side,wrist}_{1,2}` | `[N,256,256,3] uint8` | camera image, BGR | `observation.images.*` | BGR→RGB; H.264 video | dataset card + payload schema/sample | yes, H.264 video encoding |
| `obs/{side,wrist}_{1,2}_depth` | `[N,256,256] uint16` | depth image; physical unit unspecified | `observation.depth.*` | exact ArrayND Parquet storage | dataset card + payload schema/sample | no |
| `obs/tcp_pose`, `obs/tcp_vel` | `[N,7]`, `[N,6]` float64 | base-frame TCP XYZ/quaternion pose; XYZ/RPY velocity; units unspecified | matching `observation.*` | none | dataset card + payload schema | no |
| `obs/tcp_force`, `obs/tcp_torque` | `[N,3]` float64 | end-effector-frame XYZ force / RPY torque; units unspecified | matching `observation.*` | none | dataset card + payload schema | no |
| `obs/q`, `obs/dq` | `[N,7]` float64 | joint position / velocity; units unspecified | matching `observation.*` | none | dataset card + payload schema | no |
| `obs/jacobian` | `[N,6,7]` float64 | robot Jacobian | `observation.jacobian` | none | payload schema | no |
| `obs/gripper_pose` | `[N]` int64 | binary gripper state | `observation.gripper_pose` | scalar shape only | payload schema | no |
| `action` (legacy exports may use `actions`) | `[N,7]` float64 | commanded Cartesian XYZ/RPY/gripper action; units unspecified | `action` | plural alias canonicalized; otherwise none | official dataset README + payload schema | no |
| `primitive` | `[N]` Unicode string | per-frame primitive label | `observation.primitive` | UTF-8 string feature | payload schema/sample | no |
| `object_id` (multi-object) | `[N]` int64 | selected object ID | `observation.object_id` | none | payload schema/sample | no |
| `object_info` (single-object) | dictionary | length/size/shape/color/angle/distractor attributes | manifest episode provenance | preserved as JSON metadata; filename is retained as fallback provenance | official dataset README + filename convention + archive members | no |
| filename/path and board/object ID | filename tokens / path | episode task instruction | `task` and manifest `instruction` | official loader's shape/board-color mapping | official TFDS builder + filename/path | no |

Single-object and multi-object trajectories have incompatible optional schemas,
so they are separate LeRobot partitions.  A schema fingerprint creates an
additional partition if a future release changes camera/vector fields.  It
deliberately ignores each episode's frame count and NumPy's fixed Unicode
storage width because those are not logical feature-schema changes.  No episode
is concatenated across archives; an archive is the checkpoint unit and the
default local work unit contains at most eight episodes.

## Safety and resume

The preflight cache, UID-scoped work units, checkpoints, logs, encoder temporary files,
and upload queue are below
`/home/pai/zxw/functional_manipulation_benchmark_fmb_staging/`.  The
coordinator checks current usage plus the estimated peak of active units and the
next dispatch before admitting work, and enforces `--max-local-temp-bytes` and
`--min-local-free-bytes`.
Workers receive coordinator-assigned episode/task/frame ranges and final chunk
paths.  Direct commit copies Parquet/MP4 sequentially to the final staging
path, checks size, Parquet footer or video header, and first/middle/last byte
samples, then deletes local bulk.  It never uses `Path.replace` between source
and final roots.  A verified unit marker is the resume boundary; a missing
`_SUCCESS` means the collection is incomplete.

Hugging Face/Arrow intermediates are isolated under
`cache/runtime/units/<partition>/unit-XXXXXX/`.  The uploader removes that
unit cache only after the remote bulk files pass validation and the committed
marker has been written.  A failed or interrupted upload leaves the unit
cache available for retry; resume does not require it and can rebuild it.  On
startup, the converter also removes the legacy shared
`cache/runtime/datasets/` cache from older runs.  `resume/`, preflight
metadata, local unit outputs, and final OSS objects are not part of this
cleanup.

## Commands

Preflight (one archive index/schema scan; no final output):

```bash
cd /home/pai/zxw/vla-data-pipeline/embodied_datasets/scripts/convert_scripts
python3 convert_fmb_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/functional_manipulation_benchmark_fmb_extracted \
  --inspect-only
```

One bounded CPU encoder warmup is available with --warmup-frames 30. It writes
one local-only sample, validates the encoded videos, and deletes the sample:

    python3 convert_fmb_to_lerobot.py \
      --raw-root /mnt/data/embodied_datasets/public_datasets_raw/functional_manipulation_benchmark_fmb_extracted \
      --warmup-frames 30 \
      --local-work-root /home/pai/zxw/functional_manipulation_benchmark_fmb_staging \
      --encoder-threads-per-worker 8

Minimal smoke after a successful preflight, using an independent output UID
(at most one episode per selected partition):

```bash
python3 convert_fmb_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/functional_manipulation_benchmark_fmb_extracted \
  --resume --max-episodes 1 \
  --output-dataset-uid functional_manipulation_benchmark_fmb_smoke \
  --workers 1 --upload-workers 1
```

Recovery after interruption or a failed upload (the same unit markers and
local sources are reused):

```bash
python3 convert_fmb_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/functional_manipulation_benchmark_fmb_extracted \
  --resume \
  --local-work-root /home/pai/zxw/functional_manipulation_benchmark_fmb_staging \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0
```

The required 1-vs-4 worker comparison uses two independent smoke UIDs and the
same `--encoder-threads-per-worker 8`; compare the emitted `peak_local_bytes`,
worker completion order, and wall-clock/frames-per-second records before
selecting the production setting. Each run also records wall_seconds and
frames_per_second; pass --benchmark-report to write a separate JSON report:

```bash
python3 convert_fmb_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/functional_manipulation_benchmark_fmb_extracted \
  --resume --max-episodes 2 --episodes-per-unit 1 \
  --output-dataset-uid functional_manipulation_benchmark_fmb_bench_w1 --workers 1 --max-inflight-units 1 \
  --encoder-threads-per-worker 8 \
  --benchmark-report /home/pai/zxw/functional_manipulation_benchmark_fmb_staging/bench-w1.json
python3 convert_fmb_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/functional_manipulation_benchmark_fmb_extracted \
  --resume --max-episodes 2 --episodes-per-unit 1 \
  --output-dataset-uid functional_manipulation_benchmark_fmb_bench_w4 --workers 4 --max-inflight-units 4 \
  --encoder-threads-per-worker 8 \
  --benchmark-report /home/pai/zxw/functional_manipulation_benchmark_fmb_staging/bench-w4.json
```

The bounded real-data check used two read-only-copied members from valid
`multi_object_manipulation_assembly_2.zip`, with `--episodes-per-unit 1` so
both worker counts processed the same two independent units.  It was not a
full-release conversion:

| Run | Workers × encoder threads | Episodes / frames | Wall seconds | Frames/sec | Peak local bytes | Equivalence |
|---|---:|---:|---:|---:|---:|---|
| W1 | 1 × 8 | 2 / 224 | 16.85 | 13.29 | 408,647,963 | reference |
| W4 | 4 × 8 | 2 / 224 | 18.16 | 12.34 | 262,742,467 | all checks pass |

The W4 output was semantically equivalent to W1 for schema, Parquet values,
episode/task boundaries, manifest, and all 896 decoded video frames.  The
small sample is overhead-dominated, so W1 was selected for this bounded sample;
the production command below still uses the requested W4 candidate pending a
representative full-release benchmark.  A 30-frame CPU H.264 warmup completed
in 7.41 seconds (4.05 frames/sec) and was deleted.

The same bounded output was reopened with `LeRobotDataset`: 2 episodes, 224
frames, 10 FPS, 24 features, H.264 256×256 video, and exact Parquet checks for
action, TCP pose, depth, episode/frame/task indices.  A resume replay reused
both committed units without re-encoding; generic direct-commit tests also
cover interrupted upload, failed upload retaining local bulk, corruption
rejection, and the 100 GB capacity guard.

The requested production command (provided, not executed here) is:

```bash
python3 convert_fmb_to_lerobot.py --resume \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/functional_manipulation_benchmark_fmb_extracted \
  --local-work-root /home/pai/zxw/functional_manipulation_benchmark_fmb_staging \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --max-local-temp-bytes 100000000000 --min-local-free-bytes 200000000000 \
  --max-inflight-units 1 --workers 1 --encoder-threads-per-worker 8 \
  --upload-workers 1 --ossfs-io-timeout-seconds 900 \
  --trust-preflight-source
```

For a background run, use the same command with its log redirected into the
local staging root (the command is provided, not executed here):

```bash
mkdir -p /home/pai/zxw/functional_manipulation_benchmark_fmb_staging/logs
nohup python3 convert_fmb_to_lerobot.py --resume \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/functional_manipulation_benchmark_fmb_extracted \
  --local-work-root /home/pai/zxw/functional_manipulation_benchmark_fmb_staging \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --max-local-temp-bytes 100000000000 --min-local-free-bytes 200000000000 \
  --max-inflight-units 1 --workers 1 --encoder-threads-per-worker 8 \
  --upload-workers 1 --ossfs-io-timeout-seconds 900 \
  --trust-preflight-source \
  > /home/pai/zxw/functional_manipulation_benchmark_fmb_staging/logs/production.log 2>&1 &
```

The current mounted source contains two archives without a ZIP end-of-central-
directory record; the converter rejects those archives instead of converting a
partial dataset.  The bounded valid-archive checks above do not authorize a
full conversion.  Full-release preflight, production smoke, and production
benchmark remain pending until complete source archives are supplied.  All
bounded fixture outputs, work, resume, cache, logs, and OSS-test objects were
removed; the raw source was not modified.
