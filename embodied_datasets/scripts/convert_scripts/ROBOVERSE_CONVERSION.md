# RoboVerse v2 → LeRobot v3.0

This conversion is deliberately a collection of fixed-schema LeRobot datasets,
not one padded dataset. The downloaded RoboVerse release is heterogeneous and
contains trajectories but no camera samples or physical timebase.

## Evidence and source contract

The implementation is pinned to
[`RoboVerseOrg/roboverse_data@fab63ccaaed54f413901f86edc3fa1ab77a96500`](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/tree/fab63ccaaed54f413901f86edc3fa1ab77a96500).
The accompanying evidence is the [RoboVerse paper](https://huggingface.co/papers/2504.18904),
the [project page](https://roboverseorg.github.io/), the pinned
[release dataset card](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/blob/fab63ccaaed54f413901f86edc3fa1ab77a96500/README.md),
the official RoboVerse/MetaSim trajectory-loading code—including the pinned
[`convert_traj_v1_to_v2.py` source contract](https://github.com/RoboVerseOrg/RoboVerse/blob/e9b5c6efeb665052edeb934fc3172df8b9d3c9d7/scripts/conversion/convert_traj_v1_to_v2.py)—and a read-only scan of
the downloaded files at
`/mnt/data/embodied_datasets/public_datasets_raw/roboverse`.
The pinned producer explicitly builds each v2 episode from `actions`,
`init_state`, and optional `states`, then saves `{robot.name: episodes}`; it
does not write images, timestamps, FPS, or an episode instruction. Local
`extra`/file metadata and suite-specific variants are therefore accepted only
after their real containers are inspected, not inferred from that one script.
The pinned dataset card contains project/license context but no field schema,
timebase, units, camera contract, or instruction mapping, so it is not used to
fill any of those gaps.
The pinned repository exposes simulation/replay conversion rather than an
offline policy-training loader that assigns additional meanings or units to
these released pickle fields; no training-only schema claim is invented.

The real v2 trajectory contract is:

- each `_v2.pkl` or `_v2.pkl.gz` root maps robot names to episode lists, with
  optional file-level `metadata`;
- an episode has `actions`, optional `states`, and static values such as
  `init_state`/`reset_state` and `extra`;
- named actions commonly contain `dof_pos_target`; the dictionary's stored
  joint order is retained;
- state steps map entity names to numeric fields such as `pos`, `rot`, and
  named `dof_pos`;
- the release contains no trajectory RGB arrays, video files, timestamps, or
  authoritative FPS;
- some CALVIN action and state streams have different lengths; some other
  suites omit state streams; some official BiDex `initial_state_v2.json`
  sidecars are zero bytes.

The full scan also verifies every frame's ordered schema and dtype, computes
component-wise finite minimum/maximum plus NaN, positive-infinity, and
negative-infinity counts for every valid dynamic feature, checks
compressed/uncompressed aliases by decompressed SHA-256, distinguishes JSON
sidecars from episodes, inventories other unconsumed files below `trajs/`, and
fails on unreadable trajectory data. Reported
zero-byte non-trajectory sidecars block conversion unless
`--allow-source-sidecar-issues` is explicitly supplied.

Units and world/body coordinate conventions are not declared in this release.
The converter therefore labels them as source-defined and performs no unit or
coordinate transform.

### Repaired download damage and opt-in dtype-drift handling

Full conversion still requires an explicit timebase policy. The CALVIN download
damage found by the full scan has been repaired, and the official ManiSkill
source-schema drift now has a narrowly scoped opt-in conversion policy:

- the 2026-08-18 scan found 52 local CALVIN pickles below
  `trajs/calvin/env_B_out` at zero bytes. Every
  corresponding path exists as a nonzero LFS object at the pinned Hugging Face
  revision (all 52 paths were queried). For example,
  `trajectory_env_B_7496_v2.pkl` is local 0 vs official 225,903 bytes and
  `trajectory_env_B_8268_v2.pkl` is local 0 vs official 73,517 bytes. On
  2026-08-19 all 52 official objects were downloaded to local staging, checked
  against their official LFS SHA-256, reopened as pickle, and atomically
  installed at the exact raw paths. Independent post-replacement hashing and
  pickle reads passed for 52/52 files;
- 1,003 official ManiSkill files change the dtype of named finger targets
  within a source episode: 1,001 `peg_insertion_side` files, one
  `lift_peg_upright` aggregate, and one `plug_charger` file. All 1,003 local
  files are byte-identical to the pinned LFS objects by SHA-256 (53,099,549
  bytes checked). In the hash-verified
  `trajectory-franka-155_v2.pkl` object (local SHA-256 equals the pinned
  official SHA-256), frames 0–48 store both finger targets as `float32`, while
  frames 49–155 store them as Python integers (`int64` when materialized).
  LeRobot requires a fixed feature dtype within an episode. By default these
  files remain blockers. `--allow-lossless-dtype-promotion` permits only an
  integer/bool component to be promoted to the single floating dtype already
  present in that same component, and only after every promoted value survives
  a cast-and-restore equality check. It preserves frame order, shape, component
  order, and episode boundaries. Every original dtype run is recorded in the
  part manifest, collection manifest, and source episode index. A value that is
  not exactly representable (for example `16777217` as `float32`) remains a
  hard blocker.

Stable mixed-dtype or mixed-shape named mappings are supported losslessly by
splitting them into named features. Homogeneous named singleton arrays retain
their inner source axis (for example LIBERO-90 `dof_pos` is `[9, 1]`, not
silently squeezed to `[9]`). The hard case is a single named component changing
dtype or shape over time. Machine-readable size/hash/run evidence is stored at
`/home/pai/zxw/roboverse_logs/source_integrity_evidence.json`.

A targeted read-only post-repair preflight selected exactly those 52 CALVIN
paths and exited zero: 52 source files / 52 episodes / 17,145 frames / zero
blocking issues. Evidence is at
`/home/pai/zxw/roboverse_logs/calvin_repair_preflight_summary.json` and
`/home/pai/zxw/roboverse_logs/calvin_repair_preflight.log`.

On 2026-08-19 converter v10 also completed a read-only preflight of the entire
ManiSkill suite with `--allow-lossless-dtype-promotion`: 1,014 source files,
8,171 episodes, 1,448,392 action frames, 1,015 output parts, 5,432
component/episode promotion records, and zero source issues. This includes all
1,003 previously blocking files and proves that every observed integer value is
exactly representable in its component's existing floating dtype. Evidence is
at `/home/pai/zxw/roboverse_logs/maniskill_dtype_promotion_preflight.json` and
`/home/pai/zxw/roboverse_logs/maniskill_dtype_promotion_preflight.log`.

### Post-repair v10 full preflight

The 2026-08-19 converter-v10 read-only scan completed all 10,622 candidate
source files and exited zero. It found 86,727 source episodes, 10,746,462
action frames, 5,291,564 state frames, 12,534 fixed-schema parts, 126,317
linked output episodes, and 12,957,517 output frames. All 52 repaired CALVIN
files and all 1,003 ManiSkill drift files are included. The report records
5,432 exact component/episode dtype promotions, 711 feature-inventory rows,
zero blocking issues, and zero NaN/+Inf/-Inf values. The only reported source
issues are the 20 official zero-byte BiDex static sidecars, all explicitly
nonblocking. Evidence is at
`/home/pai/zxw/roboverse_logs/preflight_v10_post_repair.json` and
`/home/pai/zxw/roboverse_logs/preflight_v10_post_repair.log`.

### Historical pre-repair full preflight

The final version-9 2026-08-18 read-only scan completed all 10,622 candidate
source files in 1 hour 30 minutes 7 seconds. It inventoried 38 JSON sidecars,
33 other auxiliary files, and one decompressed duplicate alias. Eighteen nonempty static JSON
sidecars are type-preserved; the other 20 are official zero-byte BiDex
`initial_state_v2.json` placeholders and remain explicit nonblocking issues.

At the time of that scan, the valid files contained 83,959 source episodes. Lossless stream partitioning
would produce 11,479 LeRobot parts, 123,549 output episodes, and 12,456,493
frames. Output episode count is higher because unequal action/state streams are
retained as linked action-only and state-only episodes. The scan reports 1,055
blocking trajectory files: the then-truncated 52 CALVIN files and 1,003 ManiSkill
within-episode dtype changes described above. There are no other blocking issue
kinds. The expected nonzero exit records those blockers; it does not indicate
an incomplete scan. Full JSON and progress evidence:
`/home/pai/zxw/roboverse_logs/preflight_summary.json` and
`/home/pai/zxw/roboverse_logs/preflight.log`. The report also includes source
action/state frame totals, per-suite length ranges, aligned/unequal/action-only/
state-only counts, robots, tasks, and splits. Statistics are accumulated per
fixed-schema part and merged into the report; episode arrays are not retained
in memory after their part statistics are updated. The final report contains
10,245,438 source action frames, 5,274,419 source state frames, and 710 feature
inventory rows; their aggregate NaN, positive-infinity, and negative-infinity
counts are all zero.

The preserved version-8 diagnostic took 2 hours 49 seconds and reported 1,119
blockers because its reader incorrectly rejected 64 valid LIBERO-90 files whose
named singleton components have shape `(1,)`. Version 9 preserves the resulting
`[9, 1]` and `[3, 1]` axes; the full scan accepts all 65 LIBERO-90 files / 3,250
episodes. The diagnostic log/report remain at
`/home/pai/zxw/roboverse_logs/preflight_v8.log` and
`/home/pai/zxw/roboverse_logs/preflight_v8_summary.json`.

| Suite | Parts | Output episodes | Frames |
|---|---:|---:|---:|
| CALVIN | 11,297 | 86,113 | 6,718,099 |
| GAPartManip | 2 | 15 | 3,207 |
| HumanoidBench | 3 | 3 | 1,298 |
| LIBERO | 10 | 27,050 | 3,995,116 |
| LIBERO-90 | 65 | 3,250 | 493,618 |
| ManiSkill | 12 | 5,455 | 964,513 |
| MetaWorld | 1 | 1 | 64 |
| RLBench | 85 | 1,262 | 271,854 |
| SimplerEnv | 4 | 400 | 8,724 |

## Field mapping

| Source field | Source shape/dtype | Meaning / unit | LeRobot field | Conversion | Evidence | Loss |
|---|---|---|---|---|---|---|
| `actions[t].<control>` named scalar/array map | ordered `{joint_name: scalar-or-singleton-array}`, one observed dtype/shape | Named command; `dof_pos_target` is a joint-position target; units source-defined | `action`/`action.<control>` shape `(J, *component_shape)` | Values stacked in stored key order; names and every component's source shape recorded | Official loader + real files | Structural dict→named-array representation; values/dtype/order/axes unchanged |
| mixed-dtype or mixed-shape named action/state map | ordered named components with component-specific dtype/shape | Same source command/state map | one target feature per named component | Stable differences are split without casts; opt-in dynamic integer/bool values may be exactly promoted to an existing floating dtype with original dtype runs recorded | Real ManiSkill files | Structural split, or storage dtype only when exact promotion is explicitly enabled; values and episode boundaries remain unchanged |
| `actions[t]` numeric vector | source vector shape/dtype | Source action vector; units source-defined | `action` | Identity; names only accepted when an equally wide named `init_state/reset_state` DOF order exists | Real files | None |
| `states[t].<entity>.<field>` | source vector, or ordered named scalar map with recorded dtype(s) | Entity state; coordinate frame/unit source-defined | `observation.state.<entity>.<field>` | Array identity; named maps use the same flatten/split rule as action | Official loader + real files | Only the recorded structural representation |
| state field with `null` or `{}` at every step | no numeric shape/dtype | Explicitly present but carries no numeric samples | part/episode manifest provenance | Entity, field, and `null`/`empty_mapping` kind retained; included in partition schema | Real files | Not a LeRobot frame feature; no numeric value exists |
| `states[t]` absent | — | No recorded post-step state | no state feature | Action-only part; never filled | Real files | None |
| unequal `actions` / `states` | independent lengths | Alignment is not evidenced | linked action-only and state-only episodes | Both complete streams retained and linked by `source_episode_uid` | Real CALVIN files | Structural partition only |
| `init_state`, `reset_state`, `extra`, file metadata | nested, type-dependent | Static source episode/file payload | `source_episodes.jsonl.static_source_payload` | JSON type tags preserve NumPy/Torch dtype and array shape | Official loader + real files | None |
| episode/file `task_name` | string | Source-provided task label | frame `task`, LeRobot `task_index` | Identity string; actual LeRobot index mapping recorded | Real metadata | None |
| release directory task identifier | string path component | Official grouping identifier, **not claimed to be natural language** | frame `task`, LeRobot `task_index` | Prefixed by suite; origin explicitly recorded as `official_release_path_identifier_not_natural_language` | Release layout | Semantic status recorded |
| CALVIN `ann_dict.npy` | task id → instruction aliases | Official aliases, not an episode-specific instruction choice | collection manifest only | All aliases retained; no arbitrary alias is assigned to an episode | Release sidecar | None |
| source step number | integer ordinal | Ordering only; no physical time | LeRobot-generated timestamp at FPS 1 only with opt-in | `--allow-ordinal-timebase`; otherwise conversion stops | Missing timestamps/FPS in release | Physical time unavailable and declared |
| trajectory image/video | absent | No camera observation | no image/video feature | Nothing fabricated or encoded | Full local inventory | None |

LeRobot-required `index`, `episode_index`, `frame_index`, `task_index`, and
`timestamp` are generated by LeRobot. Every generated task mapping is read back
from the written dataset rather than inferred. Known Cartesian vectors use
`x/y/z`, quaternions use `w/x/y/z`, and named DOF maps retain their exact source
names. Other source arrays use `names: null`; numeric or `joint_N` placeholders
are never invented.

For a split scalar component, LeRobot declares metadata shape `[1]` but its
Hugging Face feature adapter intentionally serializes the physical Arrow column
as a scalar `Value`; the shared Parquet validator checks this canonical
singleton convention. Source component/index/dtype remain explicit, so this is
not confused with a dropped vector dimension.

For two-or-more-dimensional arrays, Hugging Face Datasets uses its fixed-shape
`ArrayND` Arrow extension. The shared validator checks the extension's declared
shape and primitive dtype. LeRobot aggregates `meta/stats.json` ranges across
all `ArrayND` components, so validation checks that aggregate against the full
source scan while manifests retain the stricter component-wise ranges and
nonfinite counts.

Source split is retained on every episode in both the part manifest and
`source_episodes.jsonl`. CALVIN `env_D_val_out` is `validation`, the four
documented `env_*_out` groups are `train`, and releases without an explicit
split remain `unspecified` rather than guessed. A LeRobot part's generated
`info.json.splits.train` is only its local storage/index view; it is not
presented as a replacement for the source split. Because part identity includes
the source file, episodes from distinct source files or split paths are never
merged into one physical part.

### Complete representational-change inventory

- Stored numeric values and component order: no numeric value change, reorder,
  normalization, clipping, padding, resampling, or value drop. Source dtype is
  unchanged by default; the explicit exact-promotion policy changes only the
  target storage dtype and retains the original per-frame dtype runs.
- Named dictionaries: structurally stacked when homogeneous, preserving any
  singleton component axes, or split into named features when components have
  distinct stable dtypes/shapes. Original path, name, index, component shape,
  target shape, dtype, split reason, and transform are in each manifest.
- Target feature path components: punctuation is converted to underscores and
  components are lowercased; the exact original `source_path` is retained and
  any normalization collision is a hard error.
- Heterogeneity: materialized as fixed-schema parts. This changes physical
  layout, not source episode identity; the collection index links every part.
- Unequal action/state lengths: represented as two linked output episodes;
  neither stream is truncated or aligned without evidence.
- Explicit all-empty/null state fields: no numeric frame value exists to write;
  their entity/field/kind remain in episode, part, and collection provenance.
- Tasks: source `task_name` is unchanged when present. Otherwise the official
  release path identifier is namespaced by suite and explicitly marked as
  non-natural-language. LeRobot's generated `task_index` map is recorded.
- Source split: preserved separately from LeRobot's generated local `train`
  storage view. Generated row/episode/frame indices are linked to source IDs.
- Time: no physical time is available. Writing requires a declared external
  integer FPS or an explicit ordinal FPS-1 representation; no resampling occurs.
- Images/video: absent in source and output; no remux or lossy encode occurs.
- Consumer API: Parquet and `info.json` preserve source dtype. The stock
  LeRobot PyTorch adapter materializes Python floating lists as float32; this
  read-time behavior is exposed in the independent reports.

## Partition and metadata design

A part key is
`source file + robot + stream alignment + ordered feature schema`. This keeps
every LeRobot dataset fixed-schema and makes the source file a durable resume
unit. Equal-length action/state streams share their original frame index.
Unequal streams are never trimmed, padded, resampled, or spuriously aligned.

Each `parts/<part-id>` is a normal LeRobot v3 dataset with an independently
generated `meta/info.json` and `conversion_manifest.json`. The collection root
contains:

- `collection_manifest.json`: source revision/files/issues, partition rule,
  counts, time policy, duplicate aliases, type-tagged non-trajectory sidecar
  payloads, full source numeric statistics for every part, and all semantic
  changes;
- `source_episodes.jsonl`: source path/robot/index/split/task, original stream
  counts, linked output part/index, and tagged static source payload.

Each part manifest repeats the source range/nonfinite evidence for its exact
schema. Validation requires `meta/stats.json`, checks feature widths and frame
counts, and requires its finite min/max to equal the full independent source
scan before a checkpoint marker or final publication is accepted.

There are no implicit dtype casts, normalization, joint reorder, field padding,
camera features, video remuxes, or video encodes. Exact dtype promotion is
available only through its explicit opt-in flag and is fully provenance-recorded.
Consequently an encoder capability preflight and remux/codec selection are not
applicable to this release; the converter asserts that no MP4 is produced.

The module boundary is intentional:

- `readers/roboverse_v2_reader.py` parses v2 containers, validates source
  schemas, exposes source-faithful frames, and builds collection/part plans;
- `convert_roboverse_to_lerobot.py` owns RoboVerse-specific partition, missing
  timebase, stream-linking, and provenance policy;
- `evaluate_roboverse_conversion.py` independently reloads raw pickle/JSON,
  Parquet, metadata, statistics, and a real `LeRobotDataset`; it does not call
  the converter reader or its sample-validation path;
- `convert_core/lerobot_writer.py`, `convert_core/progress.py`, and
  `convert_core/checkpoint.py` supply format-independent writing, validation,
  ETA, atomic JSON/fingerprints/locks, and publication primitives. The shared
  checkpoint component is also used by the config-driven conversion path and
  existing specialized converters without changing their CLI contracts.

The RoboVerse reader is intentionally consumed directly by the dedicated
entry point rather than registered in the config-driven `READER_REGISTRY`.
That registry's `DatasetReader` contract returns one fixed-schema
`DatasetConversionPlan`; it cannot represent this release's deterministic
multi-dataset collection, unequal linked streams, source-file partitions, or
per-part checkpoint units. A RoboVerse YAML/registry stub would therefore
advertise a lossy path that does not exist. The dedicated entry point is the
required boundary for the source's complex heterogeneous partitioning, while
all format-independent writing, validation, progress, checkpoint, lock, and
publication behavior remains shared.

The RoboVerse delivery adds
`readers/roboverse_v2_reader.py`,
`convert_roboverse_to_lerobot.py`,
`evaluate_roboverse_conversion.py`,
`tests/test_roboverse_v2_reader.py`,
`tests/test_convert_roboverse_to_lerobot.py`, and this document. It updates
`convert_core/lerobot_writer.py` and `tests/test_lerobot_writer.py` for generic
fixed-shape `ArrayND` validation, and updates the repository `README.md` and
`PIPELINE_STATUS.md` with the supported entry point, evidence, and status.

## Time policy

Writing is refused by default because LeRobot requires integer FPS while this
release provides neither physical timestamps nor FPS. Two explicit policies
exist:

- `--fps-override SUITE=INTEGER` records an externally evidenced nominal FPS;
- `--allow-ordinal-timebase` maps step `t` to LeRobot timestamp `t / 1` and
  records that FPS 1 is ordinal rather than physical.

Neither option resamples data. An FPS override is not proof of source timing;
the manifest says that it was user supplied without source timestamps.

## Resume and publication safety

Resume is part-granular because a source file/schema part is independently
reopenable and verifiable. For output `<parent>/<uid>`, state lives beside the
final output:

```text
.<uid>.resume/        # completed part datasets; becomes final output
.<uid>.resume-state/  # fingerprint, atomic part markers
.<uid>.resume.lock    # non-blocking advisory lock
```

The fingerprint covers resume/converter versions, source root and pinned
revision, selected files with size/mtime, selections, episodes/tasks/splits,
ordered schemas/mappings, robot and partition rule, dataset UID, FPS policy,
and the explicit no-video configuration. A marker is written only after the
part is finalized, reopened, structurally validated, and its first/middle/last
source values compare exactly. On restart, marked parts and their manifests are
revalidated; corrupt or partial parts are deleted and rebuilt. Changed source
or configuration is rejected with the changed fingerprint sections.

Normal exceptions, Ctrl-C, and SIGTERM preserve only completed markers. After
all parts and collection metadata validate, publication is an atomic rename;
overwrite uses a temporary backup and rollback. A process killed with SIGKILL
cannot finalize an open Parquet footer, but that unmarked part is never trusted
and is rebuilt on resume. If a kill lands in the narrow interval after atomic
publication but before resume-state cleanup, rerunning with `--resume` fully
validates the final output and removes that stale state; coexistence with a
separate resume-data directory is treated as ambiguous and refused.

LeRobot may create a shared `.lerobot-datasets-cache` beside datasets while
reopening parts. It is validation scratch space outside the collection; no
cache directory is permitted inside the published collection.
The conversion path rejects any resolved `--staging-root` below `/mnt/data`;
the default and documented output remain server-local under
`/home/pai/zxw/roboverse_staging`.

## Commands

Full read-only preflight (safe to run; no staging writes):

```bash
cd /home/pai/zxw/vla-data-pipeline
mkdir -p /home/pai/zxw/roboverse_logs /home/pai/zxw/.cache/huggingface
set -o pipefail
HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_roboverse_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --source-directory roboverse --dataset-uid roboverse --inspect-only \
  --allow-lossless-dtype-promotion \
  --inspection-report /home/pai/zxw/roboverse_logs/preflight_summary.json \
  --eta-interval-seconds 10 \
  2>&1 | tee /home/pai/zxw/roboverse_logs/preflight.log
```

Deterministic small-sample selection is available through repeatable
`--suite`, exact `--task`, exact `--source-path`, `--max-source-files`,
`--max-episodes`, and exact `--part`. Task filtering is applied before the
episode limit even when the authoritative task comes from file/episode
metadata rather than the release path.

Three one-episode smokes (already verified with independent UIDs):

```bash
HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_roboverse_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --staging-root /home/pai/zxw/roboverse_staging \
  --source-directory roboverse --dataset-uid roboverse_smoke \
  --source-path trajs/rlbench/basketball_in_hoop/v2/franka_v2.pkl.gz \
  --max-episodes 1 --allow-ordinal-timebase --resume

HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_roboverse_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --staging-root /home/pai/zxw/roboverse_staging \
  --source-directory roboverse --dataset-uid roboverse_mixed_dtype_smoke \
  --source-path trajs/maniskill/draw_triangle/v2/franka_v2.pkl.gz \
  --max-episodes 1 --allow-ordinal-timebase --resume

HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_roboverse_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --staging-root /home/pai/zxw/roboverse_staging \
  --source-directory roboverse --dataset-uid roboverse_multidimensional_smoke \
  --source-path trajs/libero90/libero_90_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_traj_v2.pkl \
  --max-episodes 1 --allow-ordinal-timebase --resume
```

The RLBench smoke produced 1 episode / 144 frames, robot `franka`, 8 vector
features, float64 named 9-DOF action, ordinal FPS 1, and zero videos. The
ManiSkill smoke produced 1 action-only episode / 249 frames with seven float32
arm scalar features and two int64 finger scalar features. This is the stable
mixed-dtype case: splitting preserves each named component without a cast;
unlike the 1,003 blockers, no component changes dtype during the episode.
The LIBERO-90 smoke produced 1 aligned episode / 197 frames / 15 features and
preserved named singleton arrays as exact `[9, 1]` and `[3, 1]` source,
metadata, Parquet, and runtime shapes.

The independent evaluator reopened all three outputs with `LeRobotDataset`,
reloaded the raw pickles itself, and compared every feature through both
Parquet and the LeRobot API at RLBench frames 0/72/143, ManiSkill frames
0/124/248, and LIBERO-90 frames 0/98/196. It also
recomputed full-episode ranges/nonfinite counts and checked stats, feature
metadata/names, tasks, FPS/timestamps, provenance, zero videos, and checkpoint/
cache cleanup. Run it with:

```bash
HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python \
  embodied_datasets/scripts/convert_scripts/evaluate_roboverse_conversion.py \
  --collection-root /home/pai/zxw/roboverse_staging/lerobot_v3_0/roboverse_smoke \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw/roboverse \
  --report /home/pai/zxw/roboverse_logs/smoke_evaluation.json
```

All three v9 reports pass at
`/home/pai/zxw/roboverse_logs/smoke_evaluation.json` and
`/home/pai/zxw/roboverse_logs/mixed_dtype_smoke_evaluation.json`, and
`/home/pai/zxw/roboverse_logs/multidimensional_smoke_evaluation.json`.
`meta/info.json` and Parquet retain source dtype exactly. LeRobot 0.6.0's stock
PyTorch transform turns Python floating-point lists into `torch.float32` at
read time, even for a declared/stored float64 feature; the RLBench source
values happen to be exactly float32-representable, so sampled API values are
also exactly equal. LIBERO-90 contains general float64 values: its API samples
match the documented float32 materialization while Parquet remains bit-exact.
This consumer behavior is recorded rather than misreported as a storage cast.

Focused verification covering the reader, converter/evaluator, and shared
writer is 51 passed. The final full-repository result is 455 passed / 1 skipped
/ 4 warnings; the warnings are the existing multiprocessing `fork()`
deprecation warning in four MimicGen dtype-test cases. Python compilation and
`git diff --check` also pass.

The v10 dtype-promotion verification run has 48 passing reader/converter tests,
including exact promotion, provenance runs, real Parquet/LeRobot round-trip,
independent evaluator checks, and rejection of a `16777217 -> float32` cast.
The official hash-verified drift sample and the full ManiSkill suite preflights
also pass as described above.

The full conversion is intentionally **not started**. The 52 truncated CALVIN
files have been restored and pass their targeted preflight. Converter v10 now
implements exact, provenance-recorded handling for the within-episode ManiSkill
integer/float dtype changes. The post-repair full-source preflight passes with
zero blocking issues. With the explicit ordinal-time and sidecar policies below,
the resumable background command is:

The intended full local output is
`/home/pai/zxw/roboverse_staging/lerobot_v3_0/roboverse`; it has not been
created by this work.

```bash
nohup env HF_HOME=/home/pai/zxw/.cache/huggingface \
  .venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_roboverse_to_lerobot.py \
  --raw-root /mnt/data/embodied_datasets/public_datasets_raw \
  --staging-root /home/pai/zxw/roboverse_staging \
  --source-directory roboverse --dataset-uid roboverse \
  --allow-ordinal-timebase --allow-lossless-dtype-promotion \
  --allow-source-sidecar-issues --resume \
  --workers 4 --eta-interval-seconds 10 \
  > /home/pai/zxw/roboverse_logs/convert.log 2>&1 &
echo $! > /home/pai/zxw/roboverse_logs/convert.pid
```

All three `--allow-*` switches are explicit policy decisions, not defaults.
Replace `--allow-ordinal-timebase` with evidenced per-suite `--fps-override`
values if those become available.

Monitor or resume with:

```bash
tail -f /home/pai/zxw/roboverse_logs/convert.log
ps -fp "$(cat /home/pai/zxw/roboverse_logs/convert.pid)"
find /home/pai/zxw/roboverse_staging/lerobot_v3_0/.roboverse.resume-state/parts \
  -name '*.json' | wc -l
# After Ctrl-C/SIGTERM, rerun the exact nohup command.
```

No OSS command is executed automatically. After independent final validation,
the user may inspect the following read-only dry run:

```bash
rsync -rcn --itemize-changes \
  /home/pai/zxw/roboverse_staging/lerobot_v3_0/roboverse/ \
  /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/roboverse/
```
