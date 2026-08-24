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

| Source field | Source shape/dtype | Meaning | LeRobot field | Transform | Evidence | Lossy |
|---|---|---|---|---|---|---|
| `actions` | `[T,A] float64` | following arm+hand joint target; released terminal target retained | `action` | identity | official builder `gap=3`; all following-state pairs checked | no |
| `actions2` | `[T,E] float64` | following EEF XYZ + quaternion XYZW + hand target | `action.end_effector` | identity | official builder; all following EEF pairs checked | no |
| `obs/robot0_arm_joints` | `[T,Aarm] float64` | Panda joint position(s), source order | `observation.arm_joint_position` | identity | official HDF5 schema and robot layout | no |
| `obs/robot0_hand_joints` | `[T,H] float64` | Fin-ray command or LEAP joints | `observation.hand_joint_position` | identity | official HDF5 schema and robot layout | no |
| `obs/robot0_eef_pos` | `[T,3] float64` | XYZ metres | `observation.end_effector_position` | identity | official loader/schema | no |
| `obs/robot0_eef_quat` | `[T,4] float64` | quaternion XYZW | `observation.end_effector_quaternion_xyzw` | identity | official loader/schema | no |
| `obs/pointcloud` | `[T,10000,6] float64` | XYZ metres + RGB `[0,1]` | `observation.pointcloud` | identity | official loader/builder plus sampled payload validation | no |
| `dones` | `[T] int64` | one terminal flag per phase group | `source.done` | scalar → `[1]` | released HDF5 and full low-dimensional scan | no |
| `rewards` | `[T] float64` | released zero field | `source.reward` | scalar → `[1]` | released HDF5 and full low-dimensional scan | no |
| `states` | `[T] float64` | released zero field | `source.state` | scalar → `[1]` | released HDF5 and full low-dimensional scan | no |

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
mini-dataset at `/home/pai/zxw/arcap_staging/work/<run_id>/unit-XXXXXX`. After reopen validation, one or
two uploader threads sequentially copy each Parquet/MP4 to its preassigned OSS final path. The
uploader checks exact size, reopens the Parquet footer/physical schema or MP4 stream, and compares
SHA-256 samples from the first/middle/last byte ranges. Full remote SHA-256 is diagnostic-only.
Only after validation does it delete local bulk and move compact unit metadata under local
`resume/unit_metadata`; finalization aggregates small metadata in deterministic plan order and
never performs a second full bulk copy.

The source/selection/schema/task/FPS/robot/output/conversion fingerprint is stored under
`/home/pai/zxw/arcap_staging/resume/arcap`. Recovery revalidates remote size, format and sample
evidence; retained local sources are retransmitted, while missing/corrupt objects without a local
copy rebuild only that unit. A 250 ms per-worker RSS watchdog enforces 8 GiB. Arrow,
OpenMP and BLAS nested threading are fixed at one because ARCap has no video encoder. `flock` is
non-blocking and is only a same-host guarantee. `SIGINT`/`SIGTERM` preserve verified units;
`kill -9` cannot close a current writer, so its unmarked hidden unit is discarded on recovery.
On coordinator interruption or any worker/uploader failure, the coordinator removes hidden writer
directories and targets without a verified marker while preserving verified local upload sources.

Publication is `_INCOMPLETE -> immutable chunks/meta/manifest -> _SUCCESS -> remove
_INCOMPLETE`; no whole-directory rename is used. Existing valid `_SUCCESS` is never overwritten.
`--skip-existing` is not a marker-only shortcut: under the collection lock it checks the current
source/config fingerprint, collection rows, every partition conversion manifest, and reopens all
five LeRobot datasets before accepting the output. It then removes only crash-left work/cache
state. Coordinator status is written locally to `<run_id>.json`, newline events to
`<run_id>.coordinator.jsonl`, and unit-boundary worker events to
`<run_id>.worker-XX.jsonl`; failure/interruption records include the recovery hint.

## Storage and acceleration evidence

Measured output is estimated at 61,390,182,738 bytes (57.2 GiB). The local hard limit is
100,000,000,000 bytes across `work`, queued/active uploads, `resume`, `logs` and `cache`, with at
least 200,000,000,000 local bytes left free. A coordinator-owned ledger reserves each estimated
unit peak before dispatch and blocks until an uploader deletes verified bulk. Periodic actual-use
checks provide a second guard; an individually oversized unit is rejected for further splitting.
Inspect output now states selected logical input bytes, source-container bytes, expected and
conservative output, planned unit count, maximum estimated unit peak, and a wall-time estimate
based on the measured W4/U2 end-to-end rate.
OSSFS `df` is informational only; direct object quota operations returned 403.

The final read-only full preflight reported 111,452,967,384 source-container bytes,
111,442,756,056 selected logical input bytes, 61,390,182,738 expected output bytes, 32 units, and
a 3,997,500,000-byte maximum estimated unit peak. Its 6,954.27-second (1.93-hour) estimate is a
planning projection from the bounded 33.350 frames/s W4/U2 sample, not a completed full-run claim.

The final local→OSS benchmark used the same first four assemble phase groups (12 episodes / 1,548
frames), forced four work units, and compared every 1/2/4 conversion-worker × 1/2 uploader
combination. All six outputs were semantically equivalent to W1/U1; failures, retries, encoder
errors and OSSFS errors were zero, and all benchmark output/runtime paths were removed.

| conversion workers | upload workers | wall s | frames/s | speedup | avg CPU cores | peak sampled local bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 123.733 | 12.511 | 1.000× | 0.432 | 805,571,270 |
| 1 | 2 | 122.569 | 12.630 | 1.009× | 0.429 | 630,876,057 |
| 2 | 1 | 50.743 | 30.507 | 2.438× | 1.407 | 805,569,657 |
| 2 | 2 | 54.950 | 28.171 | 2.252× | 1.406 | 795,273,113 |
| 4 | 1 | 49.396 | 31.339 | 2.505× | 2.133 | 718,829,465 |
| 4 | 2 | 46.417 | 33.350 | 2.666× | 2.275 | 805,569,657 |

This current-code matrix used eight configured encoder threads (unused because ARCap has no
video), retained no outputs, and recorded unchanged source size/mtime. W4/U2 is the selected
bounded end-to-end configuration. The historical benchmarks below used
the predecessor OSS-direct runtime; the 32-unit result remains useful worker-scaling evidence,
but neither historical run exercised this local upload pipeline.

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
The cold result must not be used to claim acceleration.

After rate-limiting recursive OSSFS inventory to the configured 120-second storage interval, the
enhanced representative benchmark used the same first 32 phase groups (96 episodes / 11,271
frames), 32 work units, and full validation for W1/W2/W4. Token `ec850faab4` passed the gate and
cleaned every output/runtime path:

| workers | wall s | frames/s | episodes/s | speedup | efficiency | avg CPU cores | peak tree RSS bytes | peak temp bytes | userspace read chars/s | staging bytes/s | I/O wait s | failures/retries/OSS errors | semantic vs W1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 883.57 | 12.756 | 0.1087 | 1.000 | 100.0% | 0.284 | 6,446,166,016 | 5,530,019,672 | 98,996,115 | 3,345,151 | 326.17 | 0/0/0 | yes |
| 2 | 494.27 | 22.803 | 0.1942 | 1.788 | 89.4% | 0.534 | 7,701,004,288 | 5,530,018,200 | 177,849,814 | 5,979,913 | 142.46 | 0/0/0 | yes |
| 4 | 453.18 | 24.871 | 0.2118 | 1.950 | 48.7% | 0.686 | 27,764,379,648 | 5,868,747,760 | 195,913,862 | 6,522,105 | 126.81 | 0/0/0 | yes |

That predecessor benchmark selected W4 and showed the formal 32-unit shape amortizes worker
startup. Its 2.59-hour projection is historical because the runtime and publication path have
since changed; the current formal command uses W4/U2 and the 100GB local ledger above.

The four A800s all failed real 30-frame H.264 NVENC with `OpenEncodeSessionEx failed: unsupported
device`; ARCap has no video anyway, and no hardware path is claimed. The formal CPU baseline keeps
the requested eight encoder threads per worker, but the manifest marks them unused because every
ARCap partition has zero video features.

The final current-code local→OSS audit smoke converted one complete assemble phase group:
3 episodes / 393 frames. With diagnostic full remote SHA-256 enabled, shell wall time was 54 s,
publication progress elapsed time was 51.19 s, and conversion took 19.96 s (19.69 frames/s).
The final collection was 104,179,332 bytes and verified upload retained 130,914 bytes of local
checkpoint metadata before cleanup. `_SUCCESS`, final-path upload evidence, full remote/local SHA
equality and an independent `LeRobotDataset` reopen all passed. A preceding default-validation
smoke took 32.69 s and confirmed full remote SHA is disabled unless explicitly requested. All
named smoke output and local work/resume/cache/log objects were removed afterward. The five raw
HDF5 size/mtime records still match the inventory captured at task start (all mtimes remain
`2026-07-24 17:01:12 +0800`); both the current audit and an earlier bounded smoke printed
`raw inventory unchanged`. Earlier two-unit evaluation additionally compared nine frames / 540,306
source scalar values bit-exactly in Parquet; presentation-layer point-cloud float32 error was at
most `2.98e-8`.

## Commands

Inspect and estimate (does not create output/runtime directories):

```bash
.venv/bin/python embodied_datasets/scripts/convert_scripts/convert_arcap_to_lerobot.py \
  --inspect-only --estimate-storage
```

Repeat the final four-unit conversion/uploader matrix (it cleans all generated outputs):

```bash
.venv/bin/python embodied_datasets/scripts/convert_scripts/convert_arcap_to_lerobot.py \
  --partition assemble --max-phase-groups 4 --max-frames-per-unit 1 \
  --benchmark-workers 1 2 4 --benchmark-upload-workers 1 2 \
  --benchmark-report /home/pai/zxw/arcap_staging/logs/benchmark.json
```

For a formal-candidate gate, use enough units to amortize persistent-worker initialization and
optionally retain the complete metric report under local logs:

```bash
.venv/bin/python embodied_datasets/scripts/convert_scripts/convert_arcap_to_lerobot.py \
  --partition assemble --max-phase-groups 32 --max-frames-per-unit 400 \
  --benchmark-workers 1 2 4 --benchmark-upload-workers 1 2 \
  --benchmark-report /home/pai/zxw/arcap_staging/logs/representative-benchmark.json
```

Formal background command (provided only; **not executed**):

```bash
LOCAL=/home/pai/zxw/arcap_staging
mkdir -p "$LOCAL/logs/arcap"

nohup .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_arcap_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/arcap \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --local-work-root "$LOCAL" \
  --acceleration-mode parallel --workers 4 --upload-workers 2 \
  --max-inflight-units 4 \
  --encoder-threads-per-worker 8 --worker-memory-limit-bytes 8589934592 \
  --max-local-temp-bytes 100000000000 \
  --min-local-free-bytes 200000000000 \
  --storage-check-interval-seconds 30 --eta-interval-seconds 10 \
  > "$LOCAL/logs/arcap/nohup.log" 2>&1 &
echo $! > "$LOCAL/logs/arcap/pid"
```

The recovery command is identical with `--resume` added. Observe it with:

```bash
tail -f /home/pai/zxw/arcap_staging/logs/arcap/nohup.log
tail -f /home/pai/zxw/arcap_staging/logs/arcap/*.coordinator.jsonl
tail -f /home/pai/zxw/arcap_staging/logs/arcap/*.worker-*.jsonl
ps -fp "$(cat /home/pai/zxw/arcap_staging/logs/arcap/pid)"
du -sb /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/arcap \
  /home/pai/zxw/arcap_staging/{work,resume,logs,cache}
find /home/pai/zxw/arcap_staging/resume/arcap -path '*/committed/*.json' | wc -l
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
- The new local→OSS matrix passed and selected W4/U2. Changes in machine, mount, or library
  versions require rerunning that gate.
- All outputs/work/resume/logs/locks from benchmark tokens `aa023c3b22`, `2f30dc76cd`, aborted
  diagnostic token `81b91e82fe`, and passing token `ec850faab4` were removed. One older zero-byte,
  zero-file OSSFS directory-marker
  ghost remains at `.conversion_work/arcap-smoke-20260819-codex`; repeated `rmdir` reports
  `Directory not empty` even though its deepest directory contains only `.` and `..`, and direct
  object-store removal is unavailable (403). It is not conversion data, but zero residual cannot
  be claimed until the mount cache/object marker is cleared by an authorized operator.

Latest bounded verification: the complete conversion-script suite passed with
`295 passed, 1 skipped`. Full low-dimensional inspect covered all
1,975 episodes / 231,923 frames and produced the 32-unit, 61,390,182,738-byte estimate without
creating its proposed runtime directory; Python compilation and `git diff --check` passed. No
full conversion or `public_datasets` publication was performed.
