# DexMimicGen → LeRobot v3 staging

This converter streams the official DexMimicGen HDF5 release into nine
schema-preserving LeRobot v3 partitions. It does not perform cleaning,
canonical embodiment alignment, or a 128-dimensional representation, and it
does not copy or modify the raw dataset.

## Fixed OSSFS layout

The only accepted runtime root is:

```text
/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0
```

The formal run uses:

```text
dexmimicgen/                                  final-compatible partition data
.conversion_work/dexmimicgen/<run_id>/       temp files and all caches
.conversion_resume/dexmimicgen/              external checkpoint state
.conversion_logs/dexmimicgen/                JSONL logs
.conversion_locks/dexmimicgen.lock           collection-wide advisory lock
```

At startup every write path is resolved, including symlinks, and rejected if
it escapes this root or overlaps another runtime root. The converter redirects
`TMPDIR`, `TMP`, `TEMP`, `XDG_CACHE_HOME`, `HF_HOME`, `HF_DATASETS_CACHE`,
`TORCH_HOME`, `MPLCONFIGDIR`, and the local Hugging Face Datasets cache below
`.conversion_work`; Python's cached `tempfile` directory is reset explicitly
both in the coordinator and in each worker. No conversion output, cache, temp
file, or checkpoint is written to `/home`, system `/tmp`, `/dev/shm`, the raw
tree, or `public_datasets`.

OSSFS `statvfs` is informational only. `--max-staging-bytes` is enforced over
final data, work, resume state, logs, and the lock. `--max-inflight-bytes` and
`--max-inflight-units` independently bound active work. Checks run before
dispatch, at each checkpoint part, periodically while frames stream, after a
partition, and before publication. The coordinator also checks before each
frontier dispatch and while workers are running, so a capacity failure stops
new work even if it originates outside an active worker.

## Source semantics

The reviewed official code revision is
`MimicGen/dexmimicgen@940e8a1b3ad70eb1925ada6b364b197de6bb2af9`; the raw
dataset revision is `181967e10c6277a653e7c9761f3978312af3646a`.

The read-only scan found 9 HDF5 files, 9,178 episodes, 2,915,177 frames, and
9,584,955 camera-frames. The containers have different fixed schemas, so each
HDF5 is one isolated LeRobot partition:

| partition | episodes | frames | action/state | cameras |
|---|---:|---:|---:|---:|
| box_cleanup | 1,016 | 234,398 | 24/103 | 3 |
| can_sort_random | 1,020 | 322,073 | 24/112 | 3 |
| coffee | 1,014 | 326,707 | 24/113 | 3 |
| drawer_cleanup | 1,026 | 298,235 | 24/92 | 3 |
| lift_tray | 1,033 | 516,848 | 24/116 | 3 |
| pouring | 1,009 | 338,519 | 24/124 | 3 |
| threading | 1,025 | 218,858 | 14/63 | 3 |
| three_piece_assembly | 1,006 | 239,827 | 14/76 | 3 |
| transport | 1,029 | 419,712 | 14/115 | 5 |

The existing validated reader and field mapping are unchanged. Numeric dtype,
shape, order, units, task, episode boundary, and provenance are preserved.
`actions`, `action_dict/*`, simulator `states`, numeric observations, and
`datagen_info/*` remain separate fields. Source RGB `uint8` images are the only
lossy path and are encoded as CPU H.264 CRF 18, preset `fast`, `yuv420p`, with
`tune=zerolatency`. MP4 output is fragmented (`frag_keyframe`, `empty_moov`,
`default_base_moof`) so the muxer only performs sequential writes supported by
OSSFS; ordinary MP4 finalization requires seek-back and truncate operations
that this mount does not support. `zerolatency` also keeps the first decoded
frame at PTS 0. Episode MJCF is stored once by content hash as zlib-compressed
sidecars.

No source timestamp, reward, done flag, mask, or missing value is fabricated.
The partition FPS is taken from `control_freq` when present and otherwise from
the official environment/playback default. Observation quaternions are XYZW;
MuJoCo free/ball `qpos` names explicitly remain WXYZ.

## TB-scale checkpoint and publication protocol

Episodes are grouped into contiguous checkpoint parts, 64 episodes by
default. A worker finalizes and reopens each part before committing one marker.
LeRobot metadata is buffered for the whole part, and repeated `info.json` and
`stats.json` writes are deferred to the part boundary. There are no per-frame
or per-episode checkpoint marker files.

Each partition writes Parquet and MP4 chunks directly at its final collection
path. The coordinator never aggregates by copying a second dataset tree and
never publishes with a full-directory rename. Publication is:

1. Create the final directory and `_INCOMPLETE`.
2. Stream isolated partition chunks in place.
3. Reopen and validate every partition, video, manifest, and MJCF sidecar.
4. Write `collection_manifest.json`, then durable `_SUCCESS` containing its
   SHA-256.
5. Delete `_INCOMPLETE` and remove work/resume state.

An existing valid `_SUCCESS` is refused by default; `--skip-existing` accepts
it only after validation. `--overwrite` remains parse-compatible but is
intentionally rejected for in-place OSSFS publication. A crash between writing
`_SUCCESS` and deleting `_INCOMPLETE` is recovered idempotently under the
collection lock.

Resume identity covers sources, selection, schema, field/task/FPS/robot
mapping, part size, and encoder options. Checkpoints inventory final data,
video, and metadata chunks with size and SHA-256. New commits hash only new or
changed chunks; resume verifies existing chunk content. An incomplete active
part is discarded. If the newest committed chunk is missing or corrupt, the
writer rolls back through part history to the newest valid prefix and rebuilds
from there.

## Parallel settings

Workers are deterministic isolated partition processes; they never share a
LeRobot writer or target. Scheduler dispatch is bounded and stops adding work
after the first worker, OSSFS, or capacity failure. Verified parts remain
resumable, and the command exits nonzero.

The earlier local-disk Transport `demo_0` benchmark measured whole-conversion
speedup of 1.00×/1.84×/3.24× for 1/2/4 workers. The final OSSFS-compatible
path was then benchmarked on `threading/demo_0` by running one identical
196-frame source episode per worker:

| workers | aggregate frames | elapsed | aggregate throughput |
|---:|---:|---:|---:|
| 1 | 196 | 14.9405 s | 13.1187 frame/s |
| 2 | 392 | 17.0237 s | 23.0267 frame/s |
| 4 | 784 | 17.2403 s | 45.4748 frame/s |

All four local A800s failed real NVENC encoding, so NVENC is rejected. The
recommended formal setting is 4 partition workers, 4 maximum inflight units,
a 5-thread encoder budget per worker, and one actual encoder thread per camera.
For a bounded selection with fewer partitions than requested workers, the
converter reports the reduced active count instead of silently claiming full
parallelism.

## Real OSSFS verification

The final path was exercised directly on the mounted staging root, not on a
local-disk substitute. A `threading/demo_0` smoke produced 1 episode / 196
frames / 3 videos. The independent evaluator reopened the raw HDF5 and output
Parquet directly and verified all 35 vector fields with exact dtype, shape,
and value equality; episode/frame/index/task mappings and the MJCF sidecar also
matched. All videos were H.264, `yuv420p`, 20 FPS, 196 frames, with first-frame
PTS 0; minimum sampled PSNR was 36.94 dB against a 30 dB gate.

A final controlled two-episode run with one episode per checkpoint part was
terminated immediately after the first verified part, exited nonzero with
`_INCOMPLETE`, and then reused exactly 1 checkpoint unit / 196 frames under
`--resume`. It produced 377 frames and two video chunks per camera. The
strengthened independent evaluator proved 35/35 numeric and 3/3 image feature
coverage, exact numeric dtype/shape/value, exact episode/frame/index/task
mapping, partition and collection manifest equivalence, `_SUCCESS` checksum,
and MJCF reference coverage. All six H.264/yuv420p videos had the expected
196/181 frames at 20 FPS; minimum sampled PSNR was 36.80 dB. A valid `_SUCCESS`
was refused by default, while `--skip-existing` validated and accepted it.
Real CLI capacity and collection-lock conflicts both exited nonzero before
creating a target dataset. The raw HDF5 remained byte-for-byte unchanged.

The final related suite was 60 passed, covering reader/converter, storage
quota, checkpoint corruption rollback, fragmented/in-place writer behavior,
and worker failure/hard-exit dispatch stopping. Syntax compilation and
`git diff --check` also passed. Pytest's synthetic writable HDF5 fixtures used
one unique `/run` directory because OSSFS cannot implement their seek/truncate
operations; that test-only directory was deleted immediately. Conversion,
evaluation, caches, pycache, logs, and smoke data remained under staging.

## Commands

Read-only estimate (does not create runtime paths):

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=embodied_datasets/scripts/convert_scripts \
.venv/bin/python embodied_datasets/scripts/convert_scripts/convert_dexmimicgen_to_lerobot.py \
  --estimate-storage
```

Formal command, to run only after separate authorization:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=embodied_datasets/scripts/convert_scripts \
.venv/bin/python embodied_datasets/scripts/convert_scripts/convert_dexmimicgen_to_lerobot.py \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --work-dir /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/dexmimicgen/dexmimicgen \
  --resume-dir /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_resume/dexmimicgen \
  --logs-dir /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_logs/dexmimicgen \
  --temp-dir /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/dexmimicgen/dexmimicgen/temp \
  --workers 4 \
  --max-inflight-units 4 \
  --encoder-threads-per-worker 5 \
  --episodes-per-checkpoint-part 64 \
  --max-staging-bytes 68719476736 \
  --max-inflight-bytes 8589934592 \
  --storage-check-interval-seconds 30
```

After interruption, rerun the identical command with `--resume`. Evaluation is
independent:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=embodied_datasets/scripts/convert_scripts \
.venv/bin/python embodied_datasets/scripts/convert_scripts/evaluate_dexmimicgen_conversion.py \
  --collection-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/dexmimicgen \
  --samples 20 --output <report-under-staging-root>.json
```

The implementation and validation task does not start the formal conversion.
