# ARCap → LeRobot v3.0 staging

This conversion is limited to `public_datasets_raw -> public_datasets_staging/lerobot_v3_0`.
It does not clean, align embodiments, resample, normalize, or create the later 128-D
STATE/ACTION representation.

## Pinned evidence and source inventory

- Dataset: `Ericcsr/ARCap@2cadcdbf4cec4ed9bc7023959583e82c8f4f0203`
- Collection code: `Ericcsr/ARCap@8fe21e533d2af8549b8c880ff331445dc0a42dbf`
- Official builder: `j96w/DexCap@5806229f00d73e70a284314ded788859fa90b89a`
- Paper: arXiv `2410.08464`

The local files match the five official LFS sizes. SHA-256 verification is available through
`--verify-source-sha256` but is optional because rereading 110.45 GB is not part of normal
metadata preflight.

| Partition | Episodes | Frames | Phase group | Bytes | SHA-256 |
|---|---:|---:|---:|---:|---|
| assemble | 480 | 58,030 | 3 | 27,872,679,936 | `e0dcc483…7012` |
| clutter | 420 | 40,109 | 3 | 19,279,966,160 | `5318e44e…3ae` |
| clutter_users | 405 | 45,495 | 3 | 21,868,455,008 | `dcea52fc…8685` |
| open_bottle | 220 | 53,290 | 2 | 25,607,833,480 | `1c0f7b44…15de` |
| wild | 450 | 34,999 | 3 | 16,824,032,800 | `52d681e8…e11f` |

The reader scans every episode ID, schema, first dimension, `num_samples`, group total and
phase boundary. It samples nine point clouds per selected file. `--full-lowdim-scan` checks
every action/state/reward/done value and following-state relationship; the much more expensive
`--full-pointcloud-scan` is separate and explicit.

The final preflight ran `--full-lowdim-scan` over all five files successfully: all 231,923
low-dimensional frames were finite, rewards/states retained their released zero values,
non-terminal actions matched the following observations, terminal actions were left untouched,
and every phase group contained exactly one final `done=1`. A full 110.45 GB point-cloud payload
scan was not run; sampled coverage was 45 frames plus schema checks for every episode.

## Semantic mapping

All source numeric leaves are retained. Scalar source leaves are represented as LeRobot shape
`[1]`; this is a shape-only container convention, not a cast. There are no image/video fields.

| Source field | Source shape/dtype | Meaning | LeRobot field | Transform | Lossy |
|---|---|---|---|---|---|
| `actions` | `[T,A] float64` | following arm+hand joint target; released terminal target retained | `action` | identity | no |
| `actions2` | `[T,E] float64` | following EEF XYZ + quaternion XYZW + hand target | `action.end_effector` | identity | no |
| `obs/robot0_arm_joints` | `[T,Aarm] float64` | Panda joint position(s), source order | `observation.arm_joint_position` | identity | no |
| `obs/robot0_hand_joints` | `[T,H] float64` | Fin-ray command or LEAP joints | `observation.hand_joint_position` | identity | no |
| `obs/robot0_eef_pos` | `[T,3] float64` | XYZ metres | `observation.end_effector_position` | identity | no |
| `obs/robot0_eef_quat` | `[T,4] float64` | quaternion XYZW | `observation.end_effector_quaternion_xyzw` | identity | no |
| `obs/pointcloud` | `[T,10000,6] float64` | XYZ metres + RGB `[0,1]` | `observation.pointcloud` | identity | no |
| `dones` | `[T] int64` | one terminal flag per phase group | `source.done` | scalar → `[1]` | no |
| `rewards` | `[T] float64` | released zero field | `source.reward` | scalar → `[1]` | no |
| `states` | `[T] float64` | released zero field | `source.state` | scalar → `[1]` | no |

`actions2` and EEF observations are genuinely absent from `open_bottle`, so that partition has
a different fixed schema rather than fabricated zeros. `open_bottle` ordering is left Panda 7,
right Panda 7; left Fin-ray command, right LEAP 16. Fin-ray is `-1=open`, `+1=close`.

Original timestamp jitter was not published. LeRobot timestamps are therefore explicitly
synthetic `frame_index / 10`; 10 Hz follows the official 30 Hz collection and builder `gap=3`.
Raw Parquet stores all source features as Arrow double/int64. LeRobot 0.6's tensor presentation
converts the 2-D point-cloud extension to float32; this consumer behavior is measured and
reported separately by the evaluator.

Tasks are evidence-backed natural language. `wild`, `clutter`, and `clutter_users` use the
paper's “Picking and placing a tennis ball with obstacles using a dexterous LEAP hand.” The `wild`
mapping is also tied to official checkpoint path `wild_tennis_3gap_test.hdf5`. Source partitions
remain separate.

## Parallel/resume/publication design

Complete phase groups are greedily combined up to 8,000 frames, yielding about 32 work units.
Four persistent spawned workers avoid repeated LeRobot imports. Each worker owns one isolated
mini-dataset and produces final-compatible Parquet. After reopen validation, an atomic marker is
written and the coordinator moves its bulk file to a preassigned final chunk on the same mount.
Only compact episode/tasks/stats metadata is aggregated in deterministic plan order; no second
full dataset copy is produced.

The source/selection/schema/task/FPS/robot/output/conversion fingerprint is stored under
`.conversion_resume/arcap`. Recovery revalidates data and marker inventories; missing, partial,
or corrupt units are rebuilt alone. A 250 ms per-worker RSS watchdog enforces 8 GiB. Arrow,
OpenMP and BLAS nested threading are fixed at one because ARCap has no video encoder. `flock` is
non-blocking and is only a same-host guarantee. `SIGINT`/`SIGTERM` preserve verified units;
`kill -9` cannot close a current writer, so its unmarked hidden unit is discarded on recovery.

Publication is `_INCOMPLETE -> immutable chunks/meta/manifest -> _SUCCESS -> remove
_INCOMPLETE`; no whole-directory rename is used. Existing valid `_SUCCESS` is never overwritten.
`--skip-existing` is not a marker-only shortcut: under the collection lock it checks the current
source/config fingerprint, collection rows, every partition conversion manifest, and reopens all
five LeRobot datasets before accepting the output. It then removes only crash-left work/resume
state. Coordinator status is written to `<run_id>.json`, newline events to
`<run_id>.coordinator.jsonl`, and unit-boundary worker events to
`<run_id>.worker-XX.jsonl`; failure/interruption records include the recovery hint.

## Storage and acceleration evidence

Measured output is estimated at 61,390,182,738 bytes (57.2 GiB). The formal limits are 80 GiB
staging, 16 GiB inflight and four inflight units. OSSFS `df` is informational only; object quota
could not be proven because `ossutil stat` returned 403. Capacity must therefore be confirmed by
the operator before the formal run.

Earlier warm, multi-unit persistent-worker runs reached 145–147 frames/s at four workers, a
measured 3.08–3.19× speedup over the one-worker baseline and 77–80% parallel efficiency. The
two-worker candidate was positive but slower; worker RSS was 1.4–2.84 GB. This evidence reflects
the 32-unit formal shape, where each persistent process amortizes its first LeRobot import.

The enhanced benchmark was also run on a deliberately tiny cold-start case: the same first four
assemble phase groups (12 episodes / 1,548 frames), four 400-frame work units, full publication
reopen, semantic comparison, and cleanup for every candidate. Here each W4 process receives only
one unit, so imports and OSSFS cache materialization dominate. The benchmark correctly returned
nonzero because neither parallel candidate accelerated this workload:

| workers | wall s | frames/s | episodes/s | avg CPU cores | peak tree RSS bytes | peak temp bytes | userspace read chars/s | staging bytes/s | I/O wait s | failures/retries | semantic vs W1 | bulk checksum equal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | 137.02 | 11.298 | 0.0876 | 0.413 | 3,455,492,096 | 806,042,433 | 93,068,735 | 2,983,520 | 7.09 | 0/0 | yes | baseline |
| 2 | 143.20 | 10.810 | 0.0838 | 0.597 | 4,947,570,688 | 806,044,046 | 92,107,158 | 2,854,780 | 7.72 | 0/0 | yes | no |
| 4 | 200.79 | 7.710 | 0.0598 | 0.607 | 11,241,447,424 | 806,044,046 | 70,062,004 | 2,036,011 | 8.52 | 0/0 | yes | no |

FUSE reported zero or incomplete physical `read_bytes`/`write_bytes` for some runs, so the table
also gives procfs userspace character throughput and actual published bytes/s. Different Parquet
container bytes are reported, not hidden; schema, every non-video value/index, episode/task
boundary, stats within declared numerical tolerance, and semantic manifests were equivalent.
The cold result must not be used to claim acceleration. Four persistent workers remain the
formal candidate based on the representative warm multi-unit evidence, but the formal command
below remains **provided only** and must not be started unless a representative enhanced
multi-unit benchmark passes again in the execution environment.

The four A800s all failed real 30-frame H.264 NVENC with `OpenEncodeSessionEx failed: unsupported
device`; ARCap has no video anyway, and no hardware path is claimed.

The final two-unit smoke converted six assemble episodes / 779 frames. It exercised worker
failure recovery, committed-unit reuse and quota refusal. Independent evaluation sampled nine
frames and compared 540,306 source scalar values bit-exactly in Parquet; `LeRobotDataset` reopened
all data, presented point clouds as float32 with max error `2.98e-8`, and synthetic float32
timestamps had max error `1.91e-7` relative to `frame_index/10`.

## Commands

Inspect and estimate (does not create output/runtime directories):

```bash
.venv/bin/python embodied_datasets/scripts/convert_scripts/convert_arcap_to_lerobot.py \
  --inspect-only --estimate-storage
```

Repeat the exact cold four-unit benchmark above (expected to reject parallel acceleration while
still cleaning all outputs):

```bash
.venv/bin/python embodied_datasets/scripts/convert_scripts/convert_arcap_to_lerobot.py \
  --partition assemble --max-phase-groups 4 --max-frames-per-unit 400 \
  --benchmark-workers 1 2 4
```

For a formal-candidate gate, use enough units to amortize persistent-worker initialization and
optionally retain the complete metric report under staging logs:

```bash
.venv/bin/python embodied_datasets/scripts/convert_scripts/convert_arcap_to_lerobot.py \
  --partition assemble --max-phase-groups 32 --max-frames-per-unit 400 \
  --benchmark-workers 1 2 4 \
  --benchmark-report /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_logs/arcap/representative-benchmark.json
```

Formal background command (provided only; **not executed**):

```bash
ROOT=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0
RUN_ID=arcap-production-v1
WORK="$ROOT/.conversion_work/arcap/$RUN_ID"
export TMPDIR="$WORK/tmp" TMP="$WORK/tmp" TEMP="$WORK/tmp"
export XDG_CACHE_HOME="$WORK/runtime_cache/xdg"
export HF_HOME="$WORK/runtime_cache/huggingface"
export HF_DATASETS_CACHE="$WORK/runtime_cache/huggingface/datasets"
export TORCH_HOME="$WORK/runtime_cache/torch"
export MPLCONFIGDIR="$WORK/runtime_cache/matplotlib"
export VLA_DATASETS_CACHE_ROOT="$WORK/runtime_cache/datasets"
export PYTHONPYCACHEPREFIX="$WORK/runtime_cache/pycache"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$WORK/tmp" "$ROOT/.conversion_logs/arcap"

nohup .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_arcap_to_lerobot.py \
  --resume --acceleration-mode parallel --workers 4 --max-inflight-units 4 \
  --encoder-threads-per-worker 1 --worker-memory-limit-bytes 8589934592 \
  --output-root "$ROOT" --work-dir "$WORK" \
  --resume-dir "$ROOT/.conversion_resume/arcap" \
  --logs-dir "$ROOT/.conversion_logs/arcap" --temp-dir "$WORK/tmp" \
  --max-staging-bytes 85899345920 --max-inflight-bytes 17179869184 \
  --storage-check-interval-seconds 30 --eta-interval-seconds 10 \
  > "$ROOT/.conversion_logs/arcap/nohup.log" 2>&1 &
echo $! > "$ROOT/.conversion_logs/arcap/pid"
```

The identical command is the recovery command. Observe it with:

```bash
tail -f "$ROOT/.conversion_logs/arcap/nohup.log"
tail -f "$ROOT/.conversion_logs/arcap/"*.coordinator.jsonl
tail -f "$ROOT/.conversion_logs/arcap/"*.worker-*.jsonl
ps -fp "$(cat "$ROOT/.conversion_logs/arcap/pid")"
du -sb "$ROOT/arcap" "$WORK" "$ROOT/.conversion_resume/arcap"
find "$ROOT/.conversion_resume/arcap" -path '*/committed/*.json' | wc -l
```

After `_SUCCESS` exists and `_INCOMPLETE` is absent:

```bash
.venv/bin/python embodied_datasets/scripts/convert_scripts/evaluate_arcap_conversion.py
```

The formal conversion has not been started. Nothing was written to
`/mnt/data/embodied_datasets/public_datasets/`.

## Outstanding validation and cleanup risks

- The full 110.45 GB point-cloud payload scan was not run; 45 point-cloud frames plus all schemas
  were checked. Object-store quota remains unknown because direct OSS operations returned 403.
- The enhanced cold four-unit benchmark failed the speedup gate as described above. A fresh
  representative multi-unit enhanced report is required before authorizing the formal command.
- All outputs/work/resume/logs/locks from benchmark tokens `aa023c3b22` and `2f30dc76cd` were
  removed and explicitly absent afterward. One older zero-byte, zero-file OSSFS directory-marker
  ghost remains at `.conversion_work/arcap-smoke-20260819-codex`; repeated `rmdir` reports
  `Directory not empty` even though its deepest directory contains only `.` and `..`, and direct
  object-store removal is unavailable (403). It is not conversion data, but zero residual cannot
  be claimed until the mount cache/object marker is cleared by an authorized operator.

Latest bounded verification: ARCap/storage/direct-commit focused tests `22 passed`; the complete
`convert_scripts/tests` suite `283 passed, 1 skipped`; full low-dimensional inspect covered all
1,975 episodes / 231,923 frames and produced the 32-unit, 61,390,182,738-byte estimate without
creating its proposed runtime directory; Python compilation and `git diff --check` passed. No
full conversion or `public_datasets` publication was performed.
