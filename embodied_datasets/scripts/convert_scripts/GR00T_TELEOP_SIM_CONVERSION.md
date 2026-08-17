# GR00T Teleop Sim to LeRobot v3.0

This document records the evidence, mapping, output design, and commands for
`convert_gr00t_teleop_sim_to_lerobot.py`. The converter treats the mounted raw
release as read-only and atomically publishes a local collection only after
each selected part has been reopened and validated.

## Evidence and pinned references

The implementation was checked against the following official sources, not
inferred from filenames alone:

| Source | Pinned revision |
|---|---|
| NVIDIA dataset card and files | `nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim@09c6de8af50168090e7e9cc01e1ec3bce788de24` |
| Dataset paper | arXiv `2503.14734` |
| NVIDIA Isaac-GR00T loader | `376ba890cff8c9de64d71d982772a9c36185fdd7` |
| RoboCasa GR1 tabletop tasks and key converters | `4840e671596f93ca03651524b9f72ffb1aadfeff` |
| robosuite GR1/Fourier MJCF | `a071383d53568ab798eb315c0e95357911be922d` (v1.5.1) |
| LeRobot v2.0 to v2.1 migration | `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` (v0.3.3) |
| LeRobot v2.1 to v3.0 migration | installed LeRobot 0.6.0 |

Every part manifest repeats these machine-readable references. The converter
also inspected every local Parquet episode, every HDF5 trajectory link, and
the source video streams. The source is:

```text
/mnt/data/embodied_datasets/public_datasets_raw/gr00t_teleop_sim
├── HDF5/
│   └── <RoboCasaTask>.hdf5                         # 24 files
└── LeRobot/
    └── gr1_unified.<RoboCasaTask>/                 # 24 v2.0 datasets
        ├── data/chunk-NNN/episode_NNNNNN.parquet
        ├── videos/chunk-NNN/observation.images.ego_view/episode_NNNNNN.mp4
        └── meta/{info,tasks,episodes,...}
```

Each source LeRobot part has its own `train: 0:100` split and exactly 1,000
episodes. Across all parts there are 24,000 episodes, 5,820,277 frames, and
186 distinct non-empty natural-language instructions. Episode lengths range
from 26 through 2,230 frames. The 24 official part/split boundaries are kept
as independently loadable v3 datasets under one collection manifest; they are
not concatenated into a new semantic split.

## Real source schema

All 24 LeRobot parts have the same frame schema:

| Field | Source dtype and shape | Meaning/evidence |
|---|---|---|
| `observation.state` | float64 `[44]` | Absolute GR1/Fourier joint positions; source Parquet, dataset card, modality metadata, key converter |
| `action` | float64 `[44]` | Absolute GR1/Fourier joint-position commands; source Parquet and the official controller/key converter |
| `observation.images.ego_view` | video `[256,256,3]` | One virtual ego RGB camera; H.264, yuv420p, no audio/depth, 20 FPS |
| `timestamp` | float64 scalar | Episode-local simulator time |
| `next.reward` | float64 scalar | Recorded reward |
| `next.done` | bool scalar | Recorded terminal flag |
| `task_index` | int64 scalar | Legacy part label; remapped as described below |
| `annotation.human.fine_action` | int64 scalar | Existing human annotation index |
| `annotation.human.coarse_action` | int64 scalar | Existing human annotation index |
| `episode_index` | int64 scalar | Source part-local episode index |
| `index` | int64 scalar | Source part-local global frame index |

There is no independent base action, joint velocity, effort, extra camera, or
audio field in the primary LeRobot representation. Therefore the output does
not invent `action.base` or any of those features. Encoded yuv420p is a video
storage pixel format; the LeRobot/PyAV reader exposes decoded three-channel
RGB. The source dataset card says the published view was cropped/padded to
256x256; the converter preserves the published frames and does not revisit the
pre-publication camera image.

The exact state/action coordinate order is shared by both 44-D features:

| Range | Group | Exact names in order |
|---|---|---|
| 0:7 | left arm | `robot0_l_shoulder_pitch`, `robot0_l_shoulder_roll`, `robot0_l_shoulder_yaw`, `robot0_l_elbow_pitch`, `robot0_l_wrist_yaw`, `robot0_l_wrist_roll`, `robot0_l_wrist_pitch` |
| 7:13 | left Fourier hand | `gripper0_left_L_pinky_intermediate_joint`, `gripper0_left_L_ring_intermediate_joint`, `gripper0_left_L_middle_intermediate_joint`, `gripper0_left_L_index_intermediate_joint`, `gripper0_left_L_thumb_proximal_pitch_joint`, `gripper0_left_L_thumb_proximal_yaw_joint` |
| 13:19 | left leg | `robot0_l_leg_hip_roll`, `robot0_l_leg_hip_yaw`, `robot0_l_leg_hip_pitch`, `robot0_l_leg_knee_pitch`, `robot0_l_leg_ankle_pitch`, `robot0_l_leg_ankle_roll` |
| 19:22 | head | `robot0_head_yaw`, `robot0_head_roll`, `robot0_head_pitch` |
| 22:29 | right arm | `robot0_r_shoulder_pitch`, `robot0_r_shoulder_roll`, `robot0_r_shoulder_yaw`, `robot0_r_elbow_pitch`, `robot0_r_wrist_yaw`, `robot0_r_wrist_roll`, `robot0_r_wrist_pitch` |
| 29:35 | right Fourier hand | `gripper0_right_R_pinky_intermediate_joint`, `gripper0_right_R_ring_intermediate_joint`, `gripper0_right_R_middle_intermediate_joint`, `gripper0_right_R_index_intermediate_joint`, `gripper0_right_R_thumb_proximal_pitch_joint`, `gripper0_right_R_thumb_proximal_yaw_joint` |
| 35:41 | right leg | `robot0_r_leg_hip_roll`, `robot0_r_leg_hip_yaw`, `robot0_r_leg_hip_pitch`, `robot0_r_leg_knee_pitch`, `robot0_r_leg_ankle_pitch`, `robot0_r_leg_ankle_roll` |
| 41:44 | waist | `robot0_torso_waist_yaw`, `robot0_torso_waist_pitch`, `robot0_torso_waist_roll` |

The official GR1 arms-and-waist key converter inserted the leg/head groups as
zeros when the original LeRobot release was generated. Those coordinates are
already present in every source row. This migration copies them and does not
perform new zero filling. The source describes the vectors as absolute joint
positions; the MuJoCo hinge coordinates are angular (radians), but the release
does not attach a per-coordinate unit or frame table. The converter therefore
does no unit or coordinate-frame conversion and records the official names and
ordering without asserting stronger semantics than the source provides.

Timestamps increase by the recorded float64 tick
`0.04999995231628418`, rather than exact decimal `0.05`. Preflight checks a
monotonic, episode-local 20 Hz cadence with precision-aware tolerance and then
copies every timestamp unchanged.

### Secondary HDF5 representation

HDF5 is a heterogeneous, image-free secondary representation, so it is not
merged into the fixed LeRobot schema. Every one of the 24,000 demos contains:

- float64 `actions` with per-frame shape `[24]`;
- float64 `states` with one of ten per-frame widths;
- float32 `action_dict/gripper [1]`, `rel_pos [3]`, `rel_rot_6d [6]`, and
  `rel_rot_axis_angle [3]`;
- attributes `num_samples`, JSON `ep_meta` (including `lang`), and a MuJoCo
  `model_file` string.

Observed state widths and episode counts are:

| Width | 119 | 130 | 132 | 143 | 145 | 156 | 158 | 169 | 182 | 195 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Episodes | 1,527 | 1,471 | 1,473 | 3,956 | 1,553 | 4,556 | 1,447 | 3,949 | 3,070 | 998 |

For each source episode, the numeric suffix of `trajectory_id` selects
`/data/demo_<id>`. Preflight checks that link, `num_samples`, and exact equality
of `remarks` with `ep_meta.lang`; manifests record every recursive HDF5 dataset
schema seen. HDF5-only state/action/action-dictionary/model/environment fields
are explicitly not copied to v3 because combining this distinct representation
with the official 44-D LeRobot representation would change its semantics.

## Task semantics

Every source frame uses legacy `task_index=1`, whose `tasks.jsonl` value is a
CamelCase directory identifier. The authoritative natural-language instruction
is the episode's official `meta/episodes.jsonl` `remarks`, independently repeated
as HDF5 `ep_meta.lang`. Only `task_index` is changed: each distinct remark gets a
part-local v3 task index. The unused empty legacy task row remains in the task
table, while the source task, source task index, source episode, and trajectory
ID remain traceable in episode metadata and the manifest. Text is not cleaned;
for example, `PnPCupToDrawerClose` really contains both official “cup” and “can”
instructions.

## Complete source-to-target mapping

| Source | LeRobot v3 target | Operation | Basis |
|---|---|---|---|
| Parquet `observation.state` | data `observation.state` | Bit-exact copy | Source schema, modality metadata, GR1/Fourier MJCF |
| Parquet `action` | data `action` | Bit-exact copy | Source schema and absolute-action key converter |
| episode MP4 `observation.images.ego_view` | packed video with same key | H.264 packet remux; no decode/re-encode | Source stream and official LeRobot migration |
| Parquet `timestamp` | data `timestamp` | Bit-exact copy | Source values; cadence checked against 20 Hz metadata |
| Parquet `next.reward` | data `next.reward` | Bit-exact copy | Source schema |
| Parquet `next.done` | data `next.done` | Bit-exact copy | Source schema |
| episode `remarks`, verified with HDF5 `ep_meta.lang` | `tasks.parquet.task`, data `task_index`, episode `tasks` | Natural-language task table and index remap | Two official annotations agree |
| Parquet `annotation.human.fine_action` | same data key | Bit-exact copy | Source schema/modality metadata |
| Parquet `annotation.human.coarse_action` | same data key | Bit-exact copy | Source schema/modality metadata |
| Parquet `episode_index` | same data key | Bit-exact copy | Source boundary/order |
| Parquet `index` | same data key | Bit-exact copy | Source part-local frame order |
| `episodes.jsonl` length, trajectory, remarks, and other fields | `episodes.parquet`, plus source episode/task trace fields | Preserve; target task list uses verified instruction | Source episode metadata |
| `info.json` FPS, robot type, split | v3 `info.json` | Preserve | Source metadata and `embodiment.json` |
| `info.json` legacy feature descriptors | v3 `info.json.features` | Preserve dtype/shape; add proven names and v3 video keys | Local schema plus official names |
| `embodiment.json`, `modality.json`, `metadata.json`, `initial_actions.npz` | same files under `meta/` when present | Byte-for-byte copy | Distributed GR00T sidecars |
| HDF5-only fields and attributes | manifest schema/cross-check record only | Not merged | Existing LeRobot format is primary; HDF5 schema is heterogeneous |
| Source part/split boundary | one v3 part plus collection entry | Preserve all 24 boundaries | Official release structure |

The same mapping, including its basis, is serialized in each
`conversion_manifest.json`.

## Final v3 `info.json` design

Every output part is a normal LeRobot v3.0 dataset with `fps=20`,
`robot_type=GR1ArmsAndWaistFourierHands`, `splits={"train":"0:100"}`, and the
following features. Counts are computed for that actual part; the full
conversion has 1,000 episodes per part and collectively 24,000 episodes and
5,820,277 frames.

| Feature | dtype | shape | names/info |
|---|---|---|---|
| `observation.images.ego_view` | video | `[256,256,3]` | HWC; H.264/yuv420p, 20 FPS, RGB decode, no audio/depth |
| `observation.state` | float64 | `[44]` | exact 44 names and order above; feature FPS 20 |
| `action` | float64 | `[44]` | exact 44 names and order above; feature FPS 20 |
| `timestamp` | float64 | `[1]` | FPS 20 |
| `next.reward` | float64 | `[1]` | FPS 20 |
| `next.done` | bool | `[1]` | FPS 20 |
| `task_index` | int64 | `[1]` | FPS 20 |
| `annotation.human.fine_action` | int64 | `[1]` | FPS 20 |
| `annotation.human.coarse_action` | int64 | `[1]` | FPS 20 |
| `episode_index` | int64 | `[1]` | FPS 20 |
| `index` | int64 | `[1]` | FPS 20 |

LeRobot v3 `info.json` has no `total_videos` field. Each part manifest records
both the number of source episode videos and actual packed target video files;
`info.json` still records the video feature and all episode/frame/task counts.

The target structure is:

```text
<staging>/lerobot_v3_0/gr00t_teleop_sim/
├── collection_manifest.json
└── part-NNN-<source-task>/
    ├── conversion_manifest.json
    ├── data/chunk-NNN/file-NNN.parquet
    ├── videos/observation.images.ego_view/chunk-NNN/file-NNN.mp4
    └── meta/
        ├── info.json
        ├── tasks.parquet
        ├── episodes/chunk-NNN/file-NNN.parquet
        ├── stats.json
        └── copied GR00T sidecars
```

## Migration, safety, and validation

The implementation follows LeRobot's official two-stage behavior:

1. v2.0 to v2.1: compute per-episode statistics and aggregate them, including
   sampled decoded video statistics.
2. v2.1 to v3.0: pack Parquet and metadata files and concatenate episode videos
   by encoded-packet remux.

No cleaning, normalization, resampling, canonical mapping, trimming, episode
joining, dimension reduction, or newly invented values occur. Serialized
Parquet values are bit-exact except for the documented `task_index` remap.
Video packets are remuxed, not re-encoded; container layout changes but encoded
image content does not. Source boundaries become collection partitions, which
is a structural migration rather than a semantic schema partition (there is one
fixed primary schema).

Validation covers all output episode metadata, lengths, boundaries, tasks,
traceability fields, data/video references, and video durations. It also checks
part counts/features/FPS/splits/task tables, byte equality of sidecars, sampled
bit-exact serialized columns, LeRobot reader values, task lookup, and decoded
first/middle/last frames. Hugging Face may present fixed-size float64 arrays as
float32 tensors; the reader cast is checked separately from the bit-exact stored
Parquet comparison.

Conversion writes a sibling `.incomplete-<uuid>` directory. It is removed on
failure and renamed only after validation. `--overwrite` uses a rollback-capable
publication helper; existing output otherwise fails, while `--skip-existing`
returns before scanning the source.

## Verified real-data runs

On 2026-08-17, a full read-only preflight passed for all 24 parts, 24,000
episodes, and 5,820,277 frames. It checked every Parquet schema/row count,
required column, timestamp sequence, episode/global index sequence, and all
24,000 HDF5 trajectory/length/language links. First/middle/last video headers
were checked in each part. It found 211 part-local task rows when unused empty
rows and cross-part duplicates are retained.

A real two-episode `PnPCupToDrawerClose` conversion passed atomic publication,
statistics, packet remux, LeRobot reopen, stored-value checks, task lookup, and
decoded-frame comparisons. It has 454 frames and preserves the official “cup”
and “can” instructions at:

```text
/home/pai/zxw/gr00t_teleop_sim_staging/lerobot_v3_0/gr00t_teleop_sim_smoke_2ep
```

## Performance investigation

The conversion was benchmarked on the available host with 40 CPU cores,
720 GiB RAM, and four idle NVIDIA A800-SXM4-80GB GPUs. Although the installed
FFmpeg/PyAV stack exposes `h264_cuvid`, GPU decoding is not advantageous for
these small 256x256 streams. Decoding one real 229-frame source episode to a
null sink took 0.196 seconds with NVDEC (58.2x real time), versus 0.166 seconds
with one CPU decoder thread (68.9x). GPU initialization and device transfer
overhead exceed the saved decode work. Video encoding is not part of this
migration at all: packing is packet remux and took roughly 0.15 seconds per ten
episodes in the benchmark.

The useful parallelism is instead the 24 independent source parts. `--workers`
runs part conversions in isolated spawned processes. Each worker writes a
disjoint directory; the parent writes the collection manifest and atomically
publishes only after every worker validates successfully. Preflight remains
ordered and read-only. A worker failure cancels pending work and removes the
entire incomplete collection.

Real-data timings were:

| Workload | Mode | Wall time | Comparison |
|---|---|---:|---:|
| 4 parts x 10 episodes | sequential | 61.801 s | baseline |
| 4 parts x 10 episodes | four standalone processes | 21.193 s | 2.92x faster |
| 4 parts x 10 episodes | integrated `--workers 4` | 27.311 s | 2.26x end-to-end |
| 8 parts x 5 episodes | integrated `--workers 4` | 29.615 s | baseline |
| 8 parts x 5 episodes | integrated `--workers 8` | 21.812 s | 1.36x faster than 4 workers |

The integrated parallel and sequential 4x10 outputs were compared across all
40 episodes and 13,689 frames: collection manifests, `info.json`, task and
episode Parquet, and every packed data value were equal. Each output had also
passed the normal LeRobot reopen and video validation.

Use 8 workers as the conservative starting point on this host. This uses CPU
parallelism, not one worker per GPU. Higher values may help but have not been
accepted as the default because the full-run bottleneck can shift to OSS/FUSE
reads or local disk. The live ETA is the authoritative full-run estimate.

## Commands

Run from `/home/pai/zxw/vla-data-pipeline` with the project `.venv`.

Full read-only preflight (every Parquet/HDF5 episode and every video header):

```bash
.venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_gr00t_teleop_sim_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --staging-root /home/pai/zxw/gr00t_teleop_sim_staging \
  --dry-run --eta-interval-seconds 10
```

Use `--sample-video-headers` only to reduce the video-header portion to the
first/middle/last episode of each part. A two-episode real smoke run is:

```bash
.venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_gr00t_teleop_sim_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --staging-root /home/pai/zxw/gr00t_teleop_sim_staging \
  --dataset-uid gr00t_teleop_sim_smoke_2ep \
  --source-task PnPCupToDrawerClose \
  --max-episodes-per-part 2 --overwrite
```

`--max-episodes-per-part` is only for explicit smoke outputs. Start the full
background conversion and save its PID/log with:

```bash
mkdir -p /home/pai/zxw/gr00t_teleop_sim_logs
nohup .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_gr00t_teleop_sim_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --staging-root /home/pai/zxw/gr00t_teleop_sim_staging \
  --sample-video-headers --workers 8 --eta-interval-seconds 10 \
  > /home/pai/zxw/gr00t_teleop_sim_logs/convert.log 2>&1 &
echo $! > /home/pai/zxw/gr00t_teleop_sim_logs/convert.pid
```

Progress lines contain current part/episode, completed/total, throughput,
elapsed time, and ETA. Inspect the process and follow the log with:

```bash
ps -fp "$(cat /home/pai/zxw/gr00t_teleop_sim_logs/convert.pid)"
tail -f /home/pai/zxw/gr00t_teleop_sim_logs/convert.log
```

The completed full target will be:

```text
/home/pai/zxw/gr00t_teleop_sim_staging/lerobot_v3_0/gr00t_teleop_sim
```

Do not encode or stage directly on `/mnt/data`; the mounted OSS/FUSE path may
not support the filesystem operations needed for safe MP4 finalization and
atomic publication.
