#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
local_root="${HOME}/1x_world_model_dataset_staging"
output_root=/mnt/data/embodied_datasets/public_datasets_staging/lerobot_v3_0
decoder_path="${local_root}/decoders/Cosmos-0.1-Tokenizer-DV8x8x8/decoder.jit"
min_free_bytes=200000000000
log_path="$local_root/console_logs/convert-fast-local.log"

if [[ ! -f "$decoder_path" ]]; then
  echo "error: Cosmos decoder not found: $decoder_path" >&2
  exit 1
fi

available_bytes=$(df -PB1 "$local_root" | awk 'NR == 2 {print $4}')
if (( available_bytes < min_free_bytes )); then
  echo "error: insufficient local free space: available=$available_bytes required=$min_free_bytes" >&2
  exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi cannot communicate with the NVIDIA driver" >&2
  exit 1
fi

mkdir -p "$(dirname "$log_path")"
cd "$repo_root"

command=(
  env
  CUDA_VISIBLE_DEVICES=3
  PYTHONDONTWRITEBYTECODE=1
  "$local_root/smoke-venv/bin/python"
  -u
  embodied_datasets/scripts/convert_scripts/convert_1x_world_model_dataset.py
  --output-root "$output_root"
  --local-work-root "$local_root"
  --output-dataset-uid 1x_world_model_dataset
  --resume
  --v1-decoder-repo "$local_root/decoders/1Xgpt"
  --cosmos-decoder-path "$decoder_path"
  --video-codec h264
  --video-quality 18
  --video-preset fast
  --workers 1
  --encoder-threads-per-worker 8
  --decode-batch-size 8
  --v1-postprocess-device gpu
  --decoder-cpu-threads 8
  --upload-workers 2
  --max-upload-queue-units 2
  --max-local-temp-bytes 100000000000
  --min-local-free-bytes 200000000000
  --max-inflight-bytes 68719476736
  --max-inflight-units 1
  --storage-check-interval-seconds 10
  --eta-interval-seconds 10
)

"${command[@]}" 2>&1 | tee -a "$log_path"
