# 1X World Model Dataset → LeRobot v3.0

Verified through a real OSSFS smoke on 2026-08-19 against the local Hugging Face snapshot at
`/mnt/data/embodied_datasets/public_datasets_raw/1x_world_model_dataset`. The source revision is
`42e3e12fff6848b511583ba6e8afa7f82ef9014e`; the official 1Xgpt code was inspected at commit
`0069734`. The full W1+U2 bounded-local conversion was started on 2026-08-20 after exact real-sample
equivalence and publication checks passed.

## Source evidence and partitioning

The local files, dataset card, bundled unpack/decoder scripts, and official training repository agree
on 30 Hz. Full binary-size, token-range, finite-float, segment-order, shard-sum, and unreferenced-file
checks passed:

| Release/split | Frames | Episodes | Storage | Observed episode lengths |
|---|---:|---:|---|---:|
| v1.1 train | 10,832,584 | 16,652 | one MAGVIT2 `uint32[16,16]` token per frame | 226–38,420 |
| v1.1 val | 54,918 | 243 | same | 226 |
| v2.0 train | 11,254,161 | 38,846 | 100 Cosmos shards + `float32[N,25]` states | 1–29,861 |
| v2.0 val | 50,300 | 493 | one Cosmos shard + states | 1–518 |

The combined convertible plan is 39,339 v2 episodes / 11,304,461 v2 frames, plus 16,895 v1
episodes / 10,887,502 v1 frames. Cross-shard v2 segments are merged by the official segment ID;
one-frame segments are retained. All expected split files are referenced exactly once and no extra
regular files were found. Full observed ranges, including every state dimension, are persisted under
`split_summaries.numeric_ranges` in each conversion manifest. Token ranges were v1 `[0,262143]`
and v2 train `[0,63999]` / val `[0,63998]`.

There is no task or scene directory hierarchy: the authoritative hierarchy is release version →
train/validation split → segment ID, with v2 storage additionally sharded. Segment IDs are required
to be monotonic, same-ID spans at a shard boundary are merged, and a repeated split selection is
rejected before it can duplicate episodes or source-file provenance.

v1.1 and v2.0 must be separate LeRobot datasets because their robot fields and tokenizers differ.
The collection root therefore contains `v1_1/`, `v2_0/`, and `collection_manifest.json`; no zero
padding or fabricated fields are used.

`test_v2.0` is intentionally rejected. Its 450 challenge samples each contain one 17-frame Cosmos
block but 64 state rows, with no segment metadata or documented alignment. Guessing would violate
the source fidelity requirement.

## Field mapping

| Source field | Source shape/dtype | Published meaning/unit | LeRobot field | Conversion | Evidence | Lossy |
|---|---|---|---|---|---|---|
| v1 `actions/joint_pos.bin` | `[N,21]`, float32 | official 21-index EVE joint order; unit and measured/command/target role unpublished | `action.joint_position` | none | official release calls the collection raw actions; card calls this folder states/closures/etc.; local sizes | no |
| v1 `actions/neck_desired.bin` | `[N,3]`, float32 | `neck_desired`; component meaning/unit and exact command role unpublished | `action.neck_desired` | none; names left null | source path plus official high-level raw-actions wording | no |
| v1 `actions/driving_command.bin` | `[N,2]`, float32 | `driving_command`; component meaning/unit and execution semantics unpublished | `action.driving_command` | none; names left null | source path plus official high-level raw-actions wording | no |
| v1 left/right closure | `[N]`, float32 | hand closure, observed `[0,1]`; unit unpublished | separate `action.*_hand_closure` | lossless reshape to `float32[1]` for LeRobot frame input; canonical Parquet storage is scalar float32 | local files, declared LeRobot schema, and physical Parquet schema | no |
| v2 `states_{shard}.bin` | `[N,25]`, float32 | official state index table; physical units unpublished | `observation.state` | none | dataset card, unpack script, local sizes/ranges | no |
| v1 MAGVIT2 tokens | `[N,16,16]`, uint32 | source reconstruction at 30 Hz | `observation.images.head` | official decoder → RGB → H.264/yuv420p | checkpoint + 1Xgpt decoder | yes, output encode |
| v2 Cosmos tokens | `[ceil(N/17),3,32,32]`, int32 | 17-frame source reconstruction at 30 Hz | `observation.images.head` | official decoder → RGB → H.264/yuv420p | bundled decoder script + NVIDIA tokenizer | yes, output encode |

The 21 joint names and v2 25 names are copied from the official dataset card. Names are deliberately
not invented for the undocumented three neck-desired and two driving-command components. No source
instruction exists, so LeRobot's required task is the explicit synthetic placeholder
`Unspecified task; the source dataset provides no instruction.` This provenance and task-index mapping
are recorded in both manifests. Source split, segment ID, exact shard/start/end spans, source paths,
and explicit LeRobot episode/task indices are recorded per episode. Each field mapping records whether
a dtype cast or reorder occurred (none here); length-one closure representation, the synthetic task, and
the lossy video encode are likewise explicit rather than implicit. Decoder paths plus the repository
HEAD or weight-file size/mtime identity used by resume are retained as `decoder_records` in the final
partition manifest.

The token decoders reconstruct 256×256 RGB. Tokens cannot be remuxed. The generated H.264/yuv420p
is conservatively recorded as lossy even at quality 0 because chroma subsampling prevents a pixel-lossless
claim. The A800 exposes CUDA decoding but has no NVENC engine: a real FFmpeg session failed on all four
cards with `OpenEncodeSessionEx failed: unsupported device`. Software H.264 is therefore the verified
choice on this host.

Every partition has LeRobot-generated `meta/info.json`, `episodes`, `tasks`, and `stats` metadata.
The shared validator reopens the dataset and now independently asserts `codebase_version=v3.0`, robot,
FPS, feature keys/dtypes/shapes/names, video height/width/channels/FPS/codec/pixel format, episode/frame/
task counts, video-feature count, split range, and exact data/video chunk path templates. Actual MP4s
are then opened to check their frame counts, rate, resolution, and codec; the manifest records the
number and validation evidence for video files.

## Authoritative references

- Pinned [Hugging Face dataset card and files](https://huggingface.co/datasets/1x-technologies/worldmodel/tree/42e3e12fff6848b511583ba6e8afa7f82ef9014e), including the official
  `huggingface-cli download 1x-technologies/worldmodel --repo-type dataset --local-dir data` command,
  v2 shard schema, state index table, v1 schema, and bundled unpack/decoder scripts.
- Official [1Xgpt repository README at inspected commit](https://github.com/1x-technologies/1Xgpt/blob/006973446bda975b93a7f93ac6a41e33ddccc865/README.md) for EVE first-person observations,
  raw-action wording, 30 Hz source data, and MAGVIT2 16×16 → 256×256 reconstruction; official
  [`data.py`](https://github.com/1x-technologies/1Xgpt/blob/006973446bda975b93a7f93ac6a41e33ddccc865/data.py) for v1 `metadata.json`, `video.bin`, and monotonic `segment_ids.bin` loading. The action
  memmap is commented out and the supplied GENIE baseline explicitly trains only on video, so neither
  is used to invent finer per-field control semantics.
- Official [phase-1](https://www.1x.tech/discover/1x-world-model) and
  [phase-2](https://www.1x.tech/discover/1x-world-model-sampling-challenge) challenge posts.
- [GENIE paper, arXiv:2402.15391](https://arxiv.org/abs/2402.15391), which is the baseline model paper
  linked by 1X—not a dedicated publication defining this dataset. No separate peer-reviewed 1X dataset
  paper was found; the official repository/card/posts are therefore the primary dataset authorities.
- Official [NVIDIA Cosmos-Tokenizer repository at inspected commit](https://github.com/NVIDIA/Cosmos-Tokenizer/tree/3584ae752ce8ebdbe06a420bf60d7513c0e878cc). The local `decoder.jit` SHA-256 is
  `881f1f6317872fad3eeeaa1e595061aa3ee12590d14ce435ac9e9e5c883e797b`.

## Bounded local work and OSS publication

The scalable workflow separates bounded local runtime state from durable OSS output. Its default
layout is:

```text
/mnt/data/.../lerobot_v3_0/1x_world_model_dataset/          # final Parquet/MP4/metadata only
/mnt/data/.../lerobot_v3_0/.conversion_locks/...            # transient publication lock
$HOME/1x_world_model_dataset_staging/
  .conversion_work/1x_world_model_dataset/<run_id>/         # unit output and temp
  .conversion_resume/1x_world_model_dataset/                 # small durable checkpoints
  .conversion_logs/1x_world_model_dataset/                   # local audit records
  .conversion_cache/                                         # persistent decoder/model cache
```

`--local-work-root` defaults to `$HOME/1x_world_model_dataset_staging`. Startup resolves and
validates the local and OSS roots independently, rejects symlink escapes and root overlap, and rejects
writes to `public_datasets`. It explicitly redirects `TMPDIR`, `TMP`,
`TEMP`, `XDG_CACHE_HOME`, `HF_HOME`, `HF_DATASETS_CACHE`, `TORCH_HOME`, `MPLCONFIGDIR`,
`VLA_DATASETS_CACHE_ROOT`, `CUDA_CACHE_PATH`, `TORCH_EXTENSIONS_DIR`, and `NUMBA_CACHE_DIR` into the
run directory. v2 checkpoints are reader-defined shard/context units; v1 checkpoints are deterministic
128-segment batches. There is no per-frame or per-episode checkpoint file.

The fingerprint includes resume schema, source root/revision, every source file path/size/mtime,
selection, schemas, source spans, field/task mapping, FPS, robot type, partition rules, decoder and
video options, and output UID. A unit is marked only after finalize, LeRobot reopen, metadata/episode
checks, and MP4 frame/FPS/resolution checks. Restart restores the last metadata snapshot, deletes files
not in its inventory, and revalidates it. Source/config/codec changes are rejected with changed
categories. `--resume`, `--overwrite`, and `--skip-existing` are mutually exclusive. SIGINT/SIGTERM
retain the last committed unit; kill -9 can corrupt the active unit, which is discarded on restart.
Workers write and fully validate units locally, patch global Parquet indices, and generate one final
SHA-256 inventory. A bounded uploader queue then copies directly to preassigned OSS paths while the
conversion worker starts the next unit. Remote verification uses object size, first/middle/last range
hashes, and Parquet/container metadata; it never decodes a complete remote video. Local bulk is deleted
only after remote verification. No second TB-scale copy is produced. Output is
created with `_INCOMPLETE`; only full collection validation writes `_SUCCESS`, then removes
`_INCOMPLETE`. This does not depend on a complete-directory rename. A valid `_SUCCESS` refuses overwrite
unless `--overwrite` is explicit. Successful publication removes local bulk work and transient unit
metadata, retains compact local resume/audit markers, and removes the OSS lock.

All growing local state, including conversion, queued/uploading units, resume, logs, temp, and cache,
is accounted together. `--max-local-temp-bytes` defaults to exactly `100000000000`; dispatch pauses
until usage is below the bound. `--min-local-free-bytes` defaults to `200000000000`, preserving 200 GB
of filesystem headroom. The approved formal topology is W1 conversion with one or two upload threads;
U2 is used only after exact W1+U1/W1+U2 equivalence passes.

Fragmented MP4 writes use
`frag_keyframe+empty_moov+default_base_moof+negative_cts_offsets`. They are created and validated on
local storage, then sequentially uploaded to OSS; libav never writes an active MP4 through OSSFS.

The implementation boundary is deliberate: `one_x_world_model_reader.py` owns official-format parsing,
full source scans, partition planning, and decoder iteration; `convert_1x_world_model_dataset.py` owns
the heterogeneous collection and real encoder/decoder preflights; `convert_core/checkpoint.py`,
`lerobot_writer.py`, `staging.py`, `direct_commit.py`, and `progress.py` own format-neutral fingerprints,
snapshots, scoped runtime paths, direct chunk commits, locks, ETA, validation, and marker publication.
The main additions are the reader, YAML config, collection CLI,
independent evaluator, checkpoint core, focused tests, and this document; the generic CLI/writer/config/
registry plus README and pipeline status were extended without removing existing entry points.

Each worker performs one bounded preflight per source version: it decodes up to 60 real frames from the
first planned episode and encodes a 60-frame in-memory RGB sample. The preflight writes no dataset
object; its in-memory container is discarded before conversion begins.

## Parallel benchmark and formal-worker decision

The persistent `multiprocessing.Pipe` scheduler lives in `convert_core/parallel.py`; it avoids
`Queue`/`SemLock` and therefore does not depend on `/dev/shm`. Process-tree resource sampling lives in
`convert_core/performance.py`; exact LeRobot comparison lives in `convert_core/equivalence.py`.
Work units are reader checkpoint boundaries, not arbitrary episode slices. This is required for v2:
17-frame Cosmos token blocks can cross episode boundaries, and splitting a decoder-context shard at an
episode changes reconstructed pixels. The coordinator freezes episode/frame/task indices and output
paths before dispatch. Each worker has its own writer, temp root, and SHA-256 verified marker. The
coordinator commits verified chunks in plan order without concatenating or copying a second bulk
dataset. Dispatch is bounded, stops after the first worker/write/capacity failure, and resume reuses
only rehashed verified units. The inflight memory/temp estimates include worker count and committed
chunk staging.

The final real-sample benchmark used the first episode from each of four distinct v2 checkpoint shards:
4 episodes / 2,765 frames, identical source tokens and selection for W1/W2/W4, CPU libx264 CRF 18 medium,
8 encoder threads per worker, and at most 32 encoder threads total. Wall time covers source planning,
decode, encode, write, validation, ordered aggregation, and publication. `/proc` process-tree counters
cover CPU/RSS/I/O; temp peak is growth across each run's dedicated final/work/resume/log roots. The
sampler never traverses the shared TB-scale staging parent.

| Workers | Wall | Frames/s | Speedup | CPU seconds / avg cores | Peak RSS | Read chars / physical read | Physical write | Temp peak | Exact vs W1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 102.62 s | 26.94 | 1.000× | 131.60 / 1.28 | 3.06 GiB | 1.30 GiB / 0 B | 99.51 MiB | 22.68 MiB | baseline |
| 2 | 87.26 s | 31.69 | 1.176× | 170.85 / 1.96 | 5.10 GiB | 2.12 GiB / 0 B | 99.66 MiB | 23.15 MiB | **failed** |
| 4 | 73.55 s | 37.59 | 1.395× | 204.15 / 2.78 | 9.40 GiB | 3.75 GiB / 8 KiB | 99.54 MiB | 23.11 MiB | **failed** |

The source was cached, hence near-zero physical reads; `read_chars` still shows bytes requested through
the process tree. W2/W4 reduce wall time, but both fail the mandatory episode image-statistics check.
Decoded-video diagnostics found W1↔W2 changed 291/2,765 frames (maximum channel delta 35, mean absolute
delta 0.130742) and W1↔W4 changed 685/2,765 frames (maximum 36, mean 0.299968). Schema, non-image
indices/values, episode/task boundaries, and counts reach the image-stat gate, but the pixel difference
is sufficient to reject the output. Deterministic PyTorch/cuDNN/cuBLAS settings, one physical GPU, and
serialized GPU calls were also tested; the official bfloat16 Cosmos decoder remains sensitive to
process/call history. These variants did not establish exact multi-worker output and are not recommended.

Therefore `parallel_eligible=false`: formal `--workers 2` and `--workers 4` are intentionally rejected
instead of silently falling back. The only approved configuration is `--workers 1` with
`--encoder-threads-per-worker 8`. W1 was separately compared with the legacy serial path on the same real
806-frame episode: schema, every index/value, episode/task boundary, all 806 decoded video frames, and
the semantic manifest were exact. That historical report predates the OSSFS-only runtime layout. New
diagnostic reports must be written below `<root>/.conversion_logs/`; the measured conclusion remains
W1-only.

The 2026-08-20 local-work uploader benchmark used the same four v1 checkpoint samples (1,699 frames),
H.264 CRF 18 `fast`, W1, and eight encoder threads. U1 completed in 187.58 s (9.06 end-to-end fps);
U2 completed in 160.38 s (10.59 fps), a 14.5% wall-time improvement. Exact comparison passed schema,
all Parquet indices/values, episode/task boundaries, semantic manifests, and all 1,699 decoded video
frames. Both runs submitted and verified four units, used no bulk aggregation copy, and published
`_SUCCESS`. Formal conversion therefore uses W1+U2; W2/W4 remain prohibited.

The v1 decoder optimization benchmark used the same 1,699 frames. After wiring the CLI options into
the reader plan, the actual GPU-side output postprocessing (`--v1-postprocess-device gpu`) with batch
size 8 passed the complete equivalence gate for both 8 and 16 PyTorch CPU threads. Batch sizes 16 and
32 failed the episode image-statistics equivalence gate for both thread counts, so batch size 8 remains
mandatory. Batch 8/thread 8 completed in about 63 s with GPU postprocessing and about 61 s with CPU
postprocessing; the GPU path is retained as an explicit validated option, but does not provide a
measurable end-to-end speedup on this host. The CPU-thread limit is applied only when a v1 unit starts.
A corrected bounded v1 GPU W1/W2 benchmark also passed complete equivalence: 61.44 s versus 52.63 s
(17.4% faster). This does not authorize W2 for the full collection because the v2 W2 benchmark remains
rejected; the formal collection topology therefore remains W1+U2. The selected conservative v1 settings
are batch 8, CPU threads 8; GPU postprocessing may be enabled with the same equivalence result when GPU
utilization is preferred.

## Commands

Full read-only preflight (loads no decoder and writes no dataset):

```bash
cd $HOME/vla-data-pipeline
$HOME/1x_world_model_dataset_staging/smoke-venv/bin/python \
  embodied_datasets/scripts/convert_scripts/convert_1x_world_model_dataset.py \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --inspect-only
```

Verified one-episode smoke commands use independent UIDs. Replace `--version` and decoder argument as
shown; both use the approved W1 configuration:

```bash
# v1
$HOME/1x_world_model_dataset_staging/smoke-venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_1x_world_model_dataset.py \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --version v1.1 --output-dataset-uid 1x_world_model_dataset_smoke_v1 \
  --max-episodes 1 --resume \
  --v1-decoder-repo $HOME/1x_world_model_dataset_staging/decoders/1Xgpt \
  --video-codec h264 --video-quality 18 --video-preset medium \
  --workers 1 --encoder-threads-per-worker 8 \
  --max-local-temp-bytes 100000000000 --min-local-free-bytes 200000000000 \
  --max-inflight-bytes 1073741824 \
  --max-inflight-units 1 --storage-check-interval-seconds 10

# v2
$HOME/1x_world_model_dataset_staging/smoke-venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_1x_world_model_dataset.py \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --version v2.0 --output-dataset-uid 1x_world_model_dataset_smoke_v2 \
  --max-episodes 1 --resume \
  --cosmos-decoder-path $HOME/1x_world_model_dataset_staging/decoders/Cosmos-0.1-Tokenizer-DV8x8x8/decoder.jit \
  --video-codec h264 --video-quality 18 --video-preset medium \
  --workers 1 --encoder-threads-per-worker 8 \
  --max-local-temp-bytes 100000000000 --min-local-free-bytes 200000000000 \
  --max-inflight-bytes 1073741824 \
  --max-inflight-units 1 --storage-check-interval-seconds 10
```

Bounded W1/W2/W4 diagnostic benchmark (never use its W2/W4 outputs as formal data):

```bash
$HOME/1x_world_model_dataset_staging/smoke-venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_1x_world_model_dataset.py \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --version v2.0 --max-checkpoint-units 4 \
  --output-dataset-uid 1x_world_model_parallel_benchmark_v2 \
  --cosmos-decoder-path $HOME/1x_world_model_dataset_staging/decoders/Cosmos-0.1-Tokenizer-DV8x8x8/decoder.jit \
  --benchmark-workers 1 2 4 --encoder-threads-per-worker 8 \
  --video-codec h264 --video-quality 18 --video-preset medium \
  --max-inflight-units 4 \
  --benchmark-report /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_logs/1x_world_model_parallel_benchmark_v2/workers.json
```

Independent evaluation must also redirect its caches into the staging root:

```bash
mkdir -p /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/tmp
env \
  TMPDIR=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/tmp \
  TMP=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/tmp \
  TEMP=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/tmp \
  XDG_CACHE_HOME=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/cache/xdg \
  HF_HOME=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/cache/huggingface \
  HF_DATASETS_CACHE=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/cache/huggingface/datasets \
  TORCH_HOME=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/cache/torch \
  MPLCONFIGDIR=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/cache/matplotlib \
  VLA_DATASETS_CACHE_ROOT=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/cache/datasets \
  CUDA_CACHE_PATH=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/cache/cuda \
  TORCH_EXTENSIONS_DIR=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/cache/torch-extensions \
  NUMBA_CACHE_DIR=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_work/1x_evaluation/cache/numba \
  $HOME/1x_world_model_dataset_staging/smoke-venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/evaluate_1x_world_model_conversion.py \
  --collection-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/1x_world_model_dataset_smoke_v1 \
  --v1-decoder-repo $HOME/1x_world_model_dataset_staging/decoders/1Xgpt \
  --output /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/.conversion_logs/1x_world_model_dataset_smoke_v1/evaluation.json
```

The authorized resumable GPU3/W1/U2 production command is:

```bash
mkdir -p $HOME/1x_world_model_dataset_staging/console_logs
env CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 \
  $HOME/1x_world_model_dataset_staging/smoke-venv/bin/python -u \
  embodied_datasets/scripts/convert_scripts/convert_1x_world_model_dataset.py \
  --output-root /mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0 \
  --local-work-root $HOME/1x_world_model_dataset_staging \
  --output-dataset-uid 1x_world_model_dataset \
  --resume \
  --v1-decoder-repo $HOME/1x_world_model_dataset_staging/decoders/1Xgpt \
  --cosmos-decoder-path $HOME/1x_world_model_dataset_staging/decoders/Cosmos-0.1-Tokenizer-DV8x8x8/decoder.jit \
  --video-codec h264 --video-quality 18 --video-preset fast \
  --workers 1 --encoder-threads-per-worker 8 \
  --decode-batch-size 8 --v1-postprocess-device gpu --decoder-cpu-threads 8 \
  --upload-workers 2 --max-upload-queue-units 2 \
  --max-local-temp-bytes 100000000000 \
  --min-local-free-bytes 200000000000 \
  --max-inflight-bytes 68719476736 \
  --max-inflight-units 1 \
  --storage-check-interval-seconds 10 \
  --eta-interval-seconds 10 2>&1 | \
  tee -a $HOME/1x_world_model_dataset_staging/console_logs/convert-fast-local.log
```

Monitor or resume with the unchanged command:

```bash
tail -f $HOME/1x_world_model_dataset_staging/console_logs/convert-fast-local.log
pgrep -af '[c]onvert_1x_world_model_dataset.py.*--output-dataset-uid 1x_world_model_dataset'
du -sh $HOME/1x_world_model_dataset_staging/.conversion_{work,resume,logs,cache}
find $HOME/1x_world_model_dataset_staging/.conversion_resume/1x_world_model_dataset/committed -type f | wc -l
```

The final output is uploaded directly to preassigned paths below
`/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0/1x_world_model_dataset`. Local work is
bounded and deleted after remote verification; there is no final rsync or second bulk copy.

## Verification results and remaining risk

- OSSFS v1 smoke (2026-08-19, GPU3/W1): 1 episode / 240 frames; six feature schemas, all 240
  `index/frame_index/episode_index/task_index/timestamp` rows, episode/task metadata, and both manifests
  passed. All five source fields were exact at first/middle/last; H.264 was 240 frames, 30 FPS,
  256×256 with a zero first PTS; sampled PSNR was 35.90–38.14 dB. Warm conversion throughput was
  27.74 frames/s; the separate cold VGG16 download took 4m49s.
- v2 smoke: 1 episode / 806 frames; state exact at first/middle/last; H.264 806 frames,
  30 FPS, 256×256; sampled PSNR 38.09–38.82 dB.
- After the append-only sink change, the previously failing v2 worker unit was resumed in place on
  GPU3 and passed its complete worker validation: 806/806 frames, H.264, 30 FPS, 256×256, first PTS 0,
  and 69.38 frames/s end-to-end for that bounded unit.
- The OSSFS smoke produced `_SUCCESS` and no `_INCOMPLETE`; work, resume, and lock paths were removed
  after publication while the audit log remained. Every recorded runtime/cache path was below the fixed
  root. `/dev/shm` stayed empty, no 1X runtime artifact appeared in `/home` or `/tmp`, and the raw
  path/size/mtime manifest remained
  `92ea28ea861c4a366a7cf834f28e60c2122f332333531cdd7c770f6f6989e544`.
- The strengthened complete `meta/info.json` contract validator passes against both existing real
  smoke partitions, not only synthetic fixtures.
- The OSSFS v1 report was generated at
  `<root>/.conversion_logs/1x_world_model_dataset_ossfs_smoke_w1/evaluation.json`; the required final
  cleanup removed that smoke UID and log after the measurements above were recorded.
- The real W1/W2/W4 v2 benchmark above rejected W2/W4 and only W1 is approved for formal conversion.
- Full-value scans found no non-finite robot values, out-of-codebook tokens, missing binaries, or
  unreferenced split files. Relevant unit/integration tests cover resume reuse/cleanup, fingerprint
  mismatch, corrupt marker, concurrent lock, and partial active-unit cleanup.
- The focused converter/reader/checkpoint/parallel/writer/performance suite, `py_compile`, and
  `git diff --check` pass; the focused suite has 60 passing tests. The repository suite excluding the
  nine localhost HTTP-server tests passed with 410 passed / 1 skipped / 4 pre-existing fork warnings.
  The unfiltered suite produced the same 410 passes but those nine unrelated tests could not call
  `socket(AF_INET)` in the managed network sandbox; the requested host-context rerun was rejected when
  the automatic approval service returned 503. This is an environment limitation, not a hidden test
  assertion failure.

Remaining uncertainties are source-side: physical units/coordinate frames are unpublished; v1
per-field measured/command/target status and neck/driving component names are unpublished; the
placeholder task is synthetic; `test_v2.0` remains
unconvertible without an official alignment; and full-scale runtime/disk size have not been measured.
There is no state/action dtype cast, reordering, normalization, resampling, cleaning, padding, or dropped
field. The only lossy transformation is decoded RGB → H.264/yuv420p; partitioning and length-one closure
representation are explicit above.
