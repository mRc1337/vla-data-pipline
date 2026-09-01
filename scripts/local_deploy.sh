#!/usr/bin/env bash
set -Eeuo pipefail

# One-command local deployment for the VLA data-governance platform.
# Usage: ./scripts/local_deploy.sh [start|stop|status|logs] [options]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${VLA_RUNTIME_DIR:-${ROOT_DIR}/.local-run}"
VENV_DIR="${VLA_VENV_DIR:-${ROOT_DIR}/.venv}"
DATA_ROOT="${VLA_DATA_ROOT:-/mnt/data/embodied_datasets/public_datasets_staging}"
CURATION_ROOT="${VLA_CURATION_ROOT:-${DATA_ROOT}/data_curation}"
CATALOG_DB="${VLA_CATALOG_DB:-${RUNTIME_DIR}/catalog.sqlite3}"
HOST="${VLA_HOST:-127.0.0.1}"
API_PORT="${VLA_API_PORT:-8000}"
WEB_PORT="${VLA_WEB_PORT:-5173}"
MODE="${VLA_DEPLOY_MODE:-dev}"
SKIP_INSTALL=0

API_PID_FILE="${RUNTIME_DIR}/api.pid"
WEB_PID_FILE="${RUNTIME_DIR}/web.pid"
API_LOG="${RUNTIME_DIR}/api.log"
WEB_LOG="${RUNTIME_DIR}/web.log"

say() { printf '[vla-deploy] %s\n' "$*"; }
die() { printf '[vla-deploy] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '1,22p' "$0"
  cat <<'EOF'

Options:
  --mode dev|docker       Deployment mode (default: dev)
  --data-root PATH        LeRobot staging root
  --curation-root PATH    Stage artifact root
  --api-port PORT         FastAPI port (default: 8000)
  --web-port PORT         Vite/Nginx port (default: 5173)
  --skip-install          Do not install Python/npm dependencies
  --help                  Show this help

Examples:
  ./scripts/local_deploy.sh start
  ./scripts/local_deploy.sh start --skip-install
  ./scripts/local_deploy.sh start --mode docker
  ./scripts/local_deploy.sh status
  ./scripts/local_deploy.sh logs api
  ./scripts/local_deploy.sh stop
EOF
}

parse_args() {
  if [[ "${1:-}" == "" || "${1:0:1}" == "-" ]]; then
    ACTION="start"
  else
    ACTION="$1"
    shift
  fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
      --data-root) DATA_ROOT="${2:?missing value for --data-root}"; shift 2 ;;
      --curation-root) CURATION_ROOT="${2:?missing value for --curation-root}"; shift 2 ;;
      --api-port) API_PORT="${2:?missing value for --api-port}"; shift 2 ;;
      --web-port) WEB_PORT="${2:?missing value for --web-port}"; shift 2 ;;
      --skip-install) SKIP_INSTALL=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  [[ "$MODE" == dev || "$MODE" == docker ]] || die "--mode must be dev or docker"
  [[ "$ACTION" == start || "$ACTION" == stop || "$ACTION" == status || "$ACTION" == logs ]] || die "action must be start, stop, status, or logs"
}

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

pid_alive() { [[ -f "$1" ]] && kill -0 "$(<"$1")" 2>/dev/null; }

cleanup_stale_pid() {
  local file="$1"
  if [[ -f "$file" ]] && ! pid_alive "$file"; then rm -f "$file"; fi
}

check_data_root() {
  [[ -d "$DATA_ROOT" ]] || die "data root does not exist: $DATA_ROOT"
  mkdir -p "$CURATION_ROOT" "$RUNTIME_DIR"
  # SQLite journaling on FUSE/OSS mounts can fail even when touch succeeds.
  # Keep the catalog on local disk unless the caller explicitly overrides it.
  if ! (touch "${CATALOG_DB}.probe" && rm -f "${CATALOG_DB}.probe"); then
    CATALOG_DB="${RUNTIME_DIR}/catalog.sqlite3"
  fi
}

install_dev_deps() {
  [[ "$SKIP_INSTALL" == 1 ]] && return
  [[ -x "${VENV_DIR}/bin/python" ]] || {
    require_cmd python3
    say "creating virtual environment: ${VENV_DIR}"
    python3 -m venv "$VENV_DIR"
  }
  say "installing Python dependencies"
  "${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/requirements.txt"
  require_cmd npm
  if [[ ! -d "${ROOT_DIR}/frontend/node_modules" ]]; then
    say "installing frontend dependencies"
    (cd "${ROOT_DIR}/frontend" && npm install)
  fi
}

start_dev() {
  require_cmd bash
  [[ -x "${VENV_DIR}/bin/uvicorn" ]] || die "uvicorn not found; run without --skip-install first"
  require_cmd npm
  [[ -d "${ROOT_DIR}/frontend/node_modules" ]] || die "frontend dependencies missing; run without --skip-install first"
  check_data_root
  cleanup_stale_pid "$API_PID_FILE"; cleanup_stale_pid "$WEB_PID_FILE"
  if pid_alive "$API_PID_FILE" || pid_alive "$WEB_PID_FILE"; then
    say "platform is already running"; status_dev; return
  fi
  export VLA_DATA_ROOT="$DATA_ROOT" VLA_CURATION_ROOT="$CURATION_ROOT" VLA_CATALOG_DB="$CATALOG_DB"
  say "starting API at http://${HOST}:${API_PORT}"
  nohup "${VENV_DIR}/bin/uvicorn" vla_platform.api:app --host "$HOST" --port "$API_PORT" \
    >"$API_LOG" 2>&1 & echo $! >"$API_PID_FILE"
  say "starting frontend at http://${HOST}:${WEB_PORT}"
  (cd "$ROOT_DIR/frontend" && nohup npm run dev -- --host "$HOST" --port "$WEB_PORT" \
    >"$WEB_LOG" 2>&1 & echo $! >"$WEB_PID_FILE")
  sleep 1
  status_dev
}

status_dev() {
  cleanup_stale_pid "$API_PID_FILE"; cleanup_stale_pid "$WEB_PID_FILE"
  printf 'api: '; pid_alive "$API_PID_FILE" && printf 'running (pid %s)\n' "$(<"$API_PID_FILE")" || printf 'stopped\n'
  printf 'web: '; pid_alive "$WEB_PID_FILE" && printf 'running (pid %s)\n' "$(<"$WEB_PID_FILE")" || printf 'stopped\n'
}

stop_dev() {
  for file in "$API_PID_FILE" "$WEB_PID_FILE"; do
    if pid_alive "$file"; then
      kill "$(<"$file")" 2>/dev/null || true
      rm -f "$file"
    fi
  done
  say "development services stopped"
}

start_docker() {
  require_cmd docker
  docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is required for --mode docker"
  check_data_root
  say "starting Docker services"
  (cd "$ROOT_DIR" && VLA_DATA_ROOT="$DATA_ROOT" docker compose up --build -d)
  say "web: http://localhost:${WEB_PORT}  api: http://localhost:${API_PORT}/docs"
}

main() {
  parse_args "$@"
  cd "$ROOT_DIR"
  case "$ACTION:$MODE" in
    start:dev) install_dev_deps; start_dev ;;
    stop:dev) stop_dev ;;
    status:dev) status_dev ;;
    logs:dev) case "${2:-api}" in api) tail -f "$API_LOG" ;; web) tail -f "$WEB_LOG" ;; *) die 'logs target must be api or web' ;; esac ;;
    start:docker) start_docker ;;
    stop:docker) require_cmd docker; docker compose down ;;
    status:docker) require_cmd docker; docker compose ps ;;
    logs:docker) require_cmd docker; docker compose logs -f "${2:-api}" ;;
  esac
}

main "$@"
