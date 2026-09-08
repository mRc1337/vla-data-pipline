# AgiBot World → LeRobot v3.0

`convert_agibot_world_to_lerobot.py` migrates the release's LeRobot v2.1
archives without decoding or re-encoding production video. MP4 bytes are
copied unchanged. Parquet values are retained exactly except for the generated
`episode_index`, `index`, and `task_index` columns, which are deterministically
rebased for their output partition.

## Field mapping

| Source field | Shape / dtype | Meaning / unit | LeRobot field | Conversion | Evidence | Lossy |
|---|---|---|---|---|---|---|
| `observation.state` | real `[169] float32`; simulation `[183] float32` | Official robot state; component units are not declared | same | Identity | source `info.json`, Parquet schema/sample | No |
| `action` | `[40] float32` | Official action; component units are not declared | same | Identity | source `info.json`, Parquet schema/sample | No |
| `ts` | simulation `[1] uint64` | Source timestamp counter; unit is not declared | same | Identity | source `info.json`, Parquet schema/sample | No |
| `action_src_status` | simulation `[1] float32` | Official action-source status | same | Identity | source `info.json`, Parquet schema/sample | No |
| `timestamp`, `frame_index` | `[1] float32`, `[1] int64` | Episode time in seconds; zero-based source frame | same | Identity | LeRobot v2.1 metadata and contiguous probe | No |
| `episode_index`, `index`, `task_index` | `[1] int64` | Generated dataset identifiers | same | Deterministic partition-global rebase | frozen task plan and direct-commit ranges | No |
| Real RGB cameras | `[H,W,3] video`; six keys in the sampled task | RGB image stream | same | MP4 byte copy; AV1 in sampled task | source metadata plus first/middle/last frame probe | No |
| `observation.images.head_depth` | `[400,640,3] video`; PNG/gray16be stream | Depth image; physical unit and v3 quantizer parameters are not declared | same | MP4 byte copy; native bytes and depth flag retained | source `video.is_depth_map`, container probe | No |
| Simulation RGB cameras | top `[400,640,3]`, hands `[1056,1280,3]` | RGB image stream | same | MP4 byte copy; HEVC in sampled task | source metadata plus frame probe | No |
| Simulation depth-patch cameras | source-declared video shapes | Depth image; physical unit is not declared | same | Patch metadata/video paired with sibling `lite/data`; MP4 byte copy | matching episode/task metadata and absent patch data archive | No |

The complete per-feature mapping and sampled media/schema evidence are stored in
each immutable `task_plan.json` and the final `collection_manifest.json`.
Because LeRobot 0.6's reader reconstructs an encoder config while opening
existing videos, PNG depth streams record their true codec/pixel format as
`source_video.codec` / `source_video.pix_fmt`; omitting unsupported encoder
fields lets the stock reader discover the copied stream through PyAV. No
`depth_unit` or quantization parameters are invented. The stock reader opens
the copied raw PNG/gray16be depth stream, but its physical unit and
quantization remain deliberately undeclared and must not be inferred.

## Transaction and index layout

Stable task keys are the official real `family/scene/task_####` directory or
the simulation `scenario/robot/variant` leaf. Fixed output partitions are
`real-imitationlearning`, `real-richinteraction`, `simulation-lite`, and
`simulation-lite-depth-patch`; schema drift inside one partition is rejected.

Each source episode is one direct-commit unit and receives a deterministic
partition-local episode/frame/unit range. Its bulk lands in
`data/chunk-NNN/file-000.parquet` and
`videos/<key>/chunk-NNN/file-000.mp4`. No uncommitted tasks share mutable bulk.
Tasks are strictly sequential; episodes inside only the current task may run
in parallel.

The order is preflight → local build/validation → copy → remote size/container/
Parquet/sample validation → durable unit marker → immediate local unit cleanup
→ task marker. Final metadata is reduced from small unit manifests/stats, and
`_SUCCESS` is written last. A dataset without `_SUCCESS` is incomplete.

## Resume and capacity safety

The lightweight catalog stores only stable paths and stat tuples. A committed
task resumes from its local marker without source preflight or remote bulk
reads. An unfinished task validates its immutable plan and source stat tuples.
Verified local units upload without archive extraction. After a unit is copied,
remotely validated, and represented by a durable unit marker, its local bulk is
deleted immediately. A later interruption resumes from that marker and does
not rebuild or re-upload the unit. A failed or incomplete upload keeps the
local unit when it is still required for retry. Timing from every failed and
successful upload attempt is accumulated in the unit marker. `--resume` also
starts a fresh run when no prior state exists; once immutable state exists it
is required and rejects changed selection, identity, or source fingerprints.

`--min-local-free-bytes` is enforced before materialization/dispatch and while
workers run. `--max-local-inflight-bytes` defaults to 128 GiB and bounds source
materialization plus all simultaneously building/uploading local units. The
coordinator lowers conversion concurrency when the requested workers do not fit
that byte budget; a single oversized unit stops before materialization. If a
shard cannot be materialized locally inside the same budget, extraction alone
uses the approved `.conversion_work/agibot_world/<uid>/<task>/<shard>` OSS
scratch; encoding is never performed there. Empty source archives are
catalogued and cause the full run to stop when reached rather than being
skipped.

With the defaults, the converter requires approximately 128 GiB of bounded
additional working data plus the 100 GiB free-space reserve (and small
metadata/cache overhead), rather than enough space for the 875 GB task. The
current filesystem had about 694.5 GiB free after smoke cleanup. Exact
per-task values are exposed by `--inspect-only`; a unit whose individual peak
exceeds the configured 128 GiB limit is rejected before conversion.

## Commands

Two-task preflight (one episode each):

```bash
python embodied_datasets/scripts/convert_scripts/convert_agibot_world_to_lerobot.py \
  --inspect-only \
  --output-dataset-uid agibot-world-preflight \
  --task real/ImitationLearning/CommercialSpaces/task_3405 \
  --task simulation/scoop_popcorn_to_bucket/g2_swift_picker/lite \
  --max-episodes-per-task 1
```

Resume that isolated smoke through commit:

```bash
python embodied_datasets/scripts/convert_scripts/convert_agibot_world_to_lerobot.py \
  --resume \
  --output-dataset-uid agibot-world-preflight \
  --task real/ImitationLearning/CommercialSpaces/task_3405 \
  --task simulation/scoop_popcorn_to_bucket/g2_swift_picker/lite \
  --max-episodes-per-task 1
```

Formal full conversion (provided only; do not run during development):

```bash
nohup python embodied_datasets/scripts/convert_scripts/convert_agibot_world_to_lerobot.py \
  --resume \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/agibot_world \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/agibot_world \
  --local-work-root $HOME/agibot_world_staging \
  --min-local-free-bytes 107374182400 \
  --max-local-inflight-bytes 137438953472 \
  --workers 4 \
  --encoder-threads-per-worker 8 \
  > $HOME/agibot_world_staging/agibot_world.log 2>&1 &
```

Status and recovery use the same immutable state:

```bash
find $HOME/agibot_world_staging/.conversion_resume/agibot_world/tasks \
  -name commit.json -print | sort
tail -f $HOME/agibot_world_staging/agibot_world.log

# Recovery: rerun the exact formal command above, including --resume.
```

Do not point the converter at the formal staging UID with task/episode limits;
subset runs require a distinct `--output-dataset-uid`.

## Repairing the pinned raw snapshot

`repair_agibot_world_snapshot.py` pins the upstream dataset at revision
`4423752b8fa754533c0cd1e6fa3ff9e4cc690566`. Its checked-in manifest contains
the official size and LFS SHA-256 (or Git blob SHA-1 for the two non-LFS files)
for all 320 files in the current non-ReinforcementLearning scope. It marks the
102 empty archives and the one truncated archive found in the 2026-08-26
snapshot as repair objects.

The current raw mount is read-only. Run the repair against a writable snapshot
or temporarily writable source mount, then publish that verified snapshot as
the converter's raw root. A size-only preflight is safe on the current mount:

```bash
python embodied_datasets/scripts/convert_scripts/repair_agibot_world_snapshot.py \
  --root /mnt/data/embodied_datasets/public_datasets_raw/agibot_world \
  --plan-only
```

The default operation streams each of the 103 pinned objects to a resumable
same-directory partial file, verifies its official size and hash, atomically
replaces the damaged file, and only then hashes all 320 scoped files. No
dataset-sized local cache is created:

```bash
python embodied_datasets/scripts/convert_scripts/repair_agibot_world_snapshot.py \
  --root /path/to/writable/agibot_world \
  --min-free-bytes 53687091200 \
  --report $HOME/agibot_world_staging/raw-repair-report.json
```

An interrupted run resumes an adjacent `.part` file with an HTTP range request.
The largest pinned repair object is 39,109,007,073 bytes, so the additional
target-filesystem peak is at most that one partial object plus an 8 MiB memory
buffer. The original object remains untouched until its replacement is fully
verified. Use `--audit-only` to repeat the complete 320-file hash audit without
downloading.

## Measured smoke evidence (2026-08-24)

- Catalog: 35 stable tasks. The source audit found 235 real archives, of which
  102 are zero bytes; nine catalog tasks reference at least one empty archive.
- Final task-scoped preflight took 37.219 s, including 7.337 s for catalog and
  startup. `task_3405` episode 0 has 2,691 frames and 464,485,269 payload bytes;
  simulation `scoop_popcorn_to_bucket/lite` episode 0 has 648 frames and
  30,933,094 bytes. Both are 30 FPS. The first plan reads one source archive;
  the simulation plan fingerprints three split archives.
- The two-task payload smoke reopened both completed partitions with stock
  `LeRobotDataset` (2 episodes / 3,339 frames total), including the raw PNG
  depth stream. Every non-generated Parquet column compared equal, generated
  ranges were contiguous, all ten MP4s had the planned frame count/FPS/shape/
  codec, and full source/final SHA-256 values matched.
- A current low-space transaction smoke converted simulation episode 0 and
  interrupted immediately after its durable unit marker. Its 30,933,094-byte
  payload had a 98,655,670-byte remote-materialization unit peak estimate and
  a 129,588,764-byte local-materialization estimate. After the interruption,
  both local `units/` and `materialized/` contained no files, while all four
  remotely validated bulk files remained; neither the task marker nor
  `_SUCCESS` existed. The cold run took 281.339 s, including archive preflight,
  and measured 2.043 s of upload. `--resume` took 9.936 s, left the unit marker
  unchanged, created the schema-v3 task marker and `_SUCCESS`, and the resulting
  648-frame partition reopened with stock `LeRobotDataset`.
- A deliberately impossible `--min-local-free-bytes 999999999999999` stopped
  before materialization/bulk writes with zero task markers.
- Identical two-episode simulation benchmark (including warmup, conversion,
  upload, finalization): 1 worker = 18.055 s; 4 workers × 8 encoder threads =
  17.786 s (1.015×). Eight bulk files / 63,584,203 bytes were hash-identical,
  so the default remains four workers.
- Final scoped regression suite: 46 passed in 13.99 s; bytecode compilation and
  `git diff --check` also passed.

The largest complete source task has 875,015,998,405 compressed bytes, but its
size no longer determines local usage: committed units are retained remotely
and released locally. With remote materialization the additional local peak is
the bounded concurrent unit estimate; with local materialization, source bytes
and concurrent units must together remain within the default 128 GiB budget.
The task preflight records the largest single-unit and requested-worker peaks.
The 102 zero-byte source archives and one truncated archive remain an
independent upstream blocker and make a full run incomplete until the source
snapshot repair above is run on a writable source location.
