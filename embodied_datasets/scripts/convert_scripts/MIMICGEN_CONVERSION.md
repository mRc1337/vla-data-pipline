# MimicGen to LeRobot v3.0

## Scope and evidence

This converter targets the official CoRL 2023 release at Hugging Face
`amandlek/mimicgen_datasets`, revision
`33016f8a62c02334f929f2913af8fdd2a8a129e1`. The semantic references are:

- MimicGen paper, *MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations*;
- official `NVlabs/mimicgen_environments` code revision
  `72bd767c255545f462e7ccfb2731f2e5d4c1d9bb`;
- official dataset card and `mimicgen/scripts/download_datasets.py`;
- robomimic `SequenceDataset`, shape inspection, and playback code;
- the downloaded HDF5 headers and metadata themselves.

The requested source path ends in `mimicgen`, but that directory does not
exist on this server. The official files are mounted under the misspelled
directory `/mnt/data/embodied_datasets/public_datasets_raw/minicgen`. The CLI
reports this fallback and records the resolved path in its checkpoint.

The mounted release is 139 GB and contains 62 HDF5 containers:

| Category | Files | Meaning |
|---|---:|---|
| `source` | 12 | 120 human demonstrations, normally 10 per task |
| `core` | 26 | generated task/reset variants |
| `object` | 2 | Mug Cleanup object variants |
| `robot` | 16 | two tasks across Panda, Sawyer, IIWA, and UR5e |
| `large_interpolation` | 6 | generated large-interpolation variants |

Each file is a robomimic container. Episodes are `data/demo_N`; per-file
metadata is `data.attrs.env_args`; optional overlapping split lists are under
`mask/*`. Generated containers normally have 1,000 episodes. The reader checks
every episode, every leaf's first dimension, `num_samples`, the summed
`data.attrs.total`, and every mask reference.

The Hugging Face tree for the pinned revision reports exactly the same 62
HDF5 relative paths and byte sizes as local storage: no missing file, extra
HDF5 file, or size mismatch. Opening and walking every local HDF5 container
during full preflight supplies an independent corruption/schema check. Files
that recur across official categories (for example a Panda robot-transfer
variant and its core counterpart) are retained as distinct official release
entries rather than deduplicated or silently dropped.

The completed full read-only preflight counted 50,120 episodes and
14,875,672 frames. Its log is
`/home/pai/zxw/mimicgen_logs/full_preflight.log`.

One official source file has stale split metadata: `source/square.hdf5`
contains only `demo_0` through `demo_9`, but eight optional masks reference
nonexistent episodes up to `demo_199` (`20_percent`: 39 dangling,
`20_percent_train`: 35, `20_percent_valid`: 4, `50_percent`: 95,
`50_percent_train`: 86, `50_percent_valid`: 9, `train`: 172, `valid`: 18).
The valid `first_10` mask has no dangling reference. The converter preserves
all raw mask lists plus an explicit `dangling_split_references` report, maps
only members that actually exist, and does not fabricate the missing demos.
All other eleven source files have one mask and no dangling reference.

## Semantic mapping

No low-dimensional fields are concatenated. Every source leaf is preserved as
its own feature with its original dtype, trailing shape, order, and values.

| Source field | Source shape/dtype | Semantics and units | LeRobot feature | Operation | Evidence | Lossy |
|---|---|---|---|---|---|---|
| `actions` | `[T,7] float64` | normalized OSC delta position xyz, delta axis-angle xyz, gripper command; controller metadata gives 0.05 m and 0.5 rad output maxima | `action` | identity | `env_args.controller_configs`; official `action_to_target_pose` / `action_to_gripper_action` | no |
| `states` | `[T,D] float64` | flattened MuJoCo simulator state; layout is environment/robot dependent | `observation.sim_state` | identity | official robomimic playback reads this as simulator state | no |
| `rewards` | `[T] float64` | environment reward | `source.reward` | scalar represented as shape `[1]` | robomimic dataset convention | no value loss |
| `dones` | `[T] int64` | environment termination flag | `source.done` | scalar represented as shape `[1]` | robomimic dataset convention | no value loss |
| `obs/robot0_joint_pos` | `[T,J] float64` | arm joint position, radians | `observation.robot0_joint_pos` | identity; names parsed from stored MuJoCo XML | robosuite observable and local model XML | no |
| `obs/robot0_joint_vel` | `[T,J] float64` | arm joint velocity, rad/s | `observation.robot0_joint_vel` | identity | robosuite observable | no |
| `obs/robot0_gripper_qpos` | `[T,G] float64` | gripper mechanism coordinates | `observation.robot0_gripper_qpos` | identity; XML joint names retained | robosuite observable and local model XML | no |
| `obs/robot0_gripper_qvel` | `[T,G] float64` | gripper mechanism coordinate velocities | `observation.robot0_gripper_qvel` | identity | robosuite observable | no |
| `obs/robot0_eef_pos` | `[T,3] float64` | end-effector position in controller/world frame, metres | `observation.robot0_eef_pos` | identity | official environment interface | no |
| `obs/robot0_eef_quat` | `[T,4] float64` | end-effector quaternion xyzw | `observation.robot0_eef_quat` | identity | robosuite observable uses `convert_quat(..., to="xyzw")` | no |
| `obs/robot0_eef_vel_lin` | `[T,3] float64` | end-effector linear velocity, m/s | same name below `observation.*` | identity | robosuite observable | no |
| `obs/robot0_eef_vel_ang` | `[T,3] float64` | end-effector angular velocity, rad/s | same name below `observation.*` | identity | robosuite observable | no |
| `obs/robot0_joint_pos_sin/cos` | `[T,J] float64` | sine/cosine of arm joint position | same name below `observation.*` | identity | stored key and robosuite observable | no |
| `obs/object` | `[T,D] float64` | official task-dependent concatenated object-state observable; component metadata is not stored in the release | `observation.object` | identity; no invented element names | official training config reads this key directly | no |
| other numeric `obs/*` | scalar/vector, native dtype | task-defined observable, including `bool` contact and force fields | same suffix below `observation.*` | identity; scalar wrapped as `[1]` | local schema and official environment implementation | no value loss |
| `obs/agentview_image` | `[T,84,84,3] uint8` | agent-view RGB | `observation.images.agentview` | streamed H.264 encoding | `env_args.camera_names`; local bytes | yes, video compression |
| `obs/robot0_eye_in_hand_image` | `[T,84,84,3] uint8` | wrist/eye-in-hand RGB | `observation.images.robot0_eye_in_hand` | streamed H.264 encoding | same | yes, video compression |

Parquet stores native doubles, integers, and booleans exactly. LeRobot 0.6's
high-level `LeRobotDataset.__getitem__` converts floating tensors to
`torch.float32` when reading; this is a LeRobot reader-view behavior, not a
conversion-time or storage cast. Exact-value evaluation therefore compares
source HDF5 with the Parquet storage and separately proves that the normal
LeRobot API can reopen the dataset.

The source has official environment IDs but no natural-language instruction.
The converter maps the twelve official environment classes to reviewed task
sentences derived from their success checks. It retains the exact environment
ID in every partition manifest. LeRobot necessarily creates `task_index`; the
generated mapping is explicit in `conversion_manifest.json`.

## Partitioning and metadata

A single LeRobot dataset requires a fixed schema. MimicGen is genuinely
heterogeneous: simulator-state and object widths differ by task; Kitchen has
optional contact/force fields; Panda/Sawyer/IIWA/UR5e joint counts and gripper
coordinates differ. One source HDF5 container is therefore one deterministic
LeRobot partition. The final layout is:

```text
/home/pai/zxw/mimicgen_staging/lerobot_v3_0/mimicgen/
├── collection_manifest.json
├── core--coffee_d0/
├── ...
└── source--three_piece_assembly/
```

Each partition has a LeRobot-generated `meta/info.json`, tasks, episodes,
stats, Parquet, videos, and `conversion_manifest.json`. The collection
manifest records aggregate counts, the partition reason, source revision,
video settings, and every source-to-output path. Overlapping `mask/*` split
membership is preserved per episode in the conversion manifest; it is not
flattened into a single fabricated split.

## Video policy

The release contains uncompressed HDF5 RGB arrays, not compatible source video
packets, so remux is impossible. The converter uses LeRobot's streaming video
encoder and never stages PNG frames. Defaults are CPU H.264, CRF 18, `fast`,
`yuv420p`, four encoder threads. Before any requested conversion it encodes and
reopens two real MimicGen frames. After each partition it uses FFprobe to check
codec, resolution, FPS, and the total frames for each camera. The current
container exposes no NVIDIA driver; `h264_nvenc` must not be used here merely
because FFmpeg lists it.

## Resume and publication

The durable unit is a complete source HDF5 partition. This matches the
release's physical shard boundary and avoids trusting an open Parquet or MP4
after power loss. Checkpoints are siblings of the final collection:

```text
.mimicgen.resume/
.mimicgen.resume-state/
.mimicgen.resume.lock
```

The fingerprint covers schema version, source repo/revision/root, source file
relative path/size/mtime, selection, every episode ID/length/split/task,
source and target schemas, FPS, robot, field/task mapping, video codec/CRF/
preset/pixel format/threads, output UID, and partition rule. A mismatch is
rejected with the changed section names. The lock is non-blocking.

A marker is atomically written only after a partition is finalized, reopened
with `LeRobotDataset`, and its videos pass FFprobe. The marker stores that
validation plus a size/mtime inventory fingerprint. The default
`--resume-validation fast` verifies unchanged checkpoints from this cheap
fingerprint and does not rescan video frames. `--resume-validation full`
forces a complete rescan. `--resume-probe-timeout 300` bounds each FFprobe;
a timeout aborts and preserves the checkpoint rather than deleting it as
corrupt. Legacy markers or changed fingerprints receive one full validation
and are upgraded atomically.

Missing, corrupt, unmarked, or `.incomplete-*` units are removed and rebuilt.
`Ctrl-C` and `SIGTERM` preserve completed partitions and discard the current
generic writer's incomplete directory. `kill -9` cannot close an active
encoder or Parquet footer; on restart the unmarked partition is discarded
rather than trusted. Before atomic publication, all partitions receive a final
cheap fingerprint check. Resume state is then removed and the lock file is
deleted. `--overwrite` uses the generic backup/rename/rollback publisher and
never deletes the valid old output first.

## Commands

Read-only full preflight:

```bash
HF_HOME=/home/pai/zxw/.cache/huggingface .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_mimicgen_to_lerobot.py \
  --inspect-only --inspect-workers 4 --eta-interval-seconds 10
```

One real smoke episode under an independent UID:

```bash
HF_HOME=/home/pai/zxw/.cache/huggingface .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_mimicgen_to_lerobot.py \
  --partition core/square_d0 --max-episodes 1 \
  --dataset-uid mimicgen_smoke_square_1ep --resume
```

Evaluate exact numeric storage plus decoded first/middle/last frame PSNR:

```bash
HF_HOME=/home/pai/zxw/.cache/huggingface .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/evaluate_mimicgen_conversion.py \
  --collection-root /home/pai/zxw/mimicgen_staging/lerobot_v3_0/mimicgen_smoke_square_1ep \
  --source-root /mnt/data/embodied_datasets/public_datasets_raw/minicgen \
  --report /home/pai/zxw/mimicgen_logs/smoke_square_1ep_evaluation.json
```

Formal background command (documented only; do not launch without explicit
authorization):

```bash
nohup env HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_mimicgen_to_lerobot.py \
  --resume --inspect-workers 4 --eta-interval-seconds 10 \
  > /home/pai/zxw/mimicgen_logs/convert.log 2>&1 &
echo $! > /home/pai/zxw/mimicgen_logs/convert.pid
```

Monitor or resume after a normal interruption:

```bash
tail -f /home/pai/zxw/mimicgen_logs/convert.log
ps -fp "$(cat /home/pai/zxw/mimicgen_logs/convert.pid)"
du -sh /home/pai/zxw/mimicgen_staging/lerobot_v3_0/.mimicgen.resume
# Resume by running the identical nohup command again.
```

Only after complete local validation, preview an OSS copy and checksum. These
commands are deliberately dry-run only:

```bash
rsync -rn --info=progress2 \
  /home/pai/zxw/mimicgen_staging/lerobot_v3_0/mimicgen/ \
  /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/mimicgen/
rsync -rcn --itemize-changes \
  /home/pai/zxw/mimicgen_staging/lerobot_v3_0/mimicgen/ \
  /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/mimicgen/
```

## Verified and outstanding

The real `core/square_d0.hdf5::demo_0` smoke output contains 1 episode / 136
frames. It reopened through `LeRobotDataset`; Parquet numeric samples were
bit-exact at frames 0, 68, and 135; six decoded camera samples had minimum
PSNR 34.99 dB; both streams passed codec/FPS/resolution/frame-count checks.
The machine-readable report is
`/home/pai/zxw/mimicgen_logs/smoke_square_1ep_evaluation.json`.

No full formal conversion has been launched, no existing output has been
overwritten, and nothing has been copied to OSS. Remaining risks are total
runtime/storage for all 48,000+ episodes and lossy H.264 image quality outside
the sampled smoke episode. The full source preflight has passed for 62
partitions / 50,120 episodes / 14,875,672 frames. The source `obs/object` and simulator-state vectors
do not carry per-element names in the official release; inventing such names
would be less faithful, so their exact vectors and top-level semantics are
preserved without fabricated element labels.
