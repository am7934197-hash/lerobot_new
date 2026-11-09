#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-fineart_vla}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
TLS_DIR="${TLS_DIR:-${PROJECT_ROOT}/secrets/s1_inference_tls}"
HOST="0.0.0.0"
PORT="18003"

case "${PROFILE}" in
  pi05)
    CHECKPOINT_PATH="/home/zhangyuqi/zhangyuqi/models/S1/10.28/run_20261028_121132_b3f637a6/checkpoints/025000/pretrained_model"
    CHECKPOINT_FINGERPRINT="6d1b810527a742f49934278636ed10f31f3ebc0ce5e9373f2dc121170da86ac8"
    ;;
  fineart_vla)
    CHECKPOINT_PATH="/home/zhangyuqi/zhangyuqi/models/S1/10.29/fineart_vla_rtc_subtask/checkpoints/030000/pretrained_model"
    CHECKPOINT_FINGERPRINT="6642caf1ba417b9757d099fb3b70a347c5fdcec9adb59c8fcec7e277a626c9df"
    ;;
  *)
    echo "未知 Profile：${PROFILE}（可选 pi05 / fineart_vla）" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
exec /home/zhangyuqi/.conda/envs/lerobot_new/bin/python -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps=30 \
  --expected_policy_type="${PROFILE}" \
  --expected_checkpoint_path="${CHECKPOINT_PATH}" \
  --expected_checkpoint_fingerprint="${CHECKPOINT_FINGERPRINT}" \
  --require_mtls=true \
  --tls_ca_cert_path="${TLS_DIR}/ca.crt" \
  --tls_server_cert_path="${TLS_DIR}/server.crt" \
  --tls_server_key_path="${TLS_DIR}/server.key"
