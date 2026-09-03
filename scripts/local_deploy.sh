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
VIDEO_PROXY_ROOT="${VLA_VIDEO_PROXY_ROOT:-${RUNTIME_DIR}/video_proxy}"
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

read_pid() {
  local file="$1" pid
  [[ -f "$file" ]] || return 1
  pid="$(<"$file")"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "$pid"
}

pid_alive() {
  local pid
  pid="$(read_pid "$1")" || return 1
  kill -0 "$pid" 2>/dev/null
}

process_group_id() { ps -o pgid= -p "$1" 2>/dev/null | tr -d '[:space:]'; }
process_session_id() { ps -o sid= -p "$1" 2>/dev/null | tr -d '[:space:]'; }

managed_process() {
  local file="$1" service="$2" pid pgid command
  pid="$(read_pid "$file")" || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  pgid="$(process_group_id "$pid")"
  [[ "$pgid" == "$pid" ]] || return 1
  command="$(ps -o args= -p "$pid" 2>/dev/null)"
  case "$service" in
    api) [[ "$command" == *"vla_platform.api:app"* ]] ;;
    web) [[ "$command" == *"frontend/node_modules/.bin/vite"* && "$command" == *"--strictPort"* ]] ;;
    *) return 1 ;;
  esac
}

cleanup_stale_pid() {
  local file="$1" service="$2"
  if [[ -f "$file" ]] && ! managed_process "$file" "$service"; then
    rm -f "$file"
  fi
}

terminate_group() {
  local file="$1" service="$2" pid attempt
  [[ -f "$file" ]] || return 0
  if ! managed_process "$file" "$service"; then
    say "ignoring stale or unsafe ${service} PID file"
    rm -f "$file"
    return 0
  fi
  pid="$(read_pid "$file")"
  kill -TERM -- "-$pid" 2>/dev/null || true
  for attempt in {1..50}; do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 -- "-$pid" 2>/dev/null; then
    say "${service} process group ${pid} did not stop; sending SIGKILL"
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  rm -f "$file"
}

wait_for_service() {
  local file="$1" service="$2" attempt pid
  for attempt in {1..20}; do
    if managed_process "$file" "$service"; then
      sleep 0.1
      continue
    fi
    pid="$(read_pid "$file")" || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 0.1
  done
  managed_process "$file" "$service"
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
  require_cmd setsid
  [[ -x "${VENV_DIR}/bin/uvicorn" ]] || die "uvicorn not found; run without --skip-install first"
  local vite_bin="${ROOT_DIR}/frontend/node_modules/.bin/vite"
  [[ -x "$vite_bin" ]] || die "Vite not found; run without --skip-install first"
  check_data_root
  cleanup_stale_pid "$API_PID_FILE" api
  cleanup_stale_pid "$WEB_PID_FILE" web
  if managed_process "$API_PID_FILE" api || managed_process "$WEB_PID_FILE" web; then
    say "platform is already running"; status_dev; return
  fi
  export VLA_DATA_ROOT="$DATA_ROOT" VLA_CURATION_ROOT="$CURATION_ROOT" VLA_CATALOG_DB="$CATALOG_DB"
  export VLA_VIDEO_PROXY_ROOT="$VIDEO_PROXY_ROOT"
  say "starting API at http://${HOST}:${API_PORT}"
  nohup setsid "${VENV_DIR}/bin/uvicorn" vla_platform.api:app --host "$HOST" --port "$API_PORT" \
    >"$API_LOG" 2>&1 &
  printf '%s\n' "$!" >"$API_PID_FILE"
  say "starting frontend at http://${HOST}:${WEB_PORT}"
  cd "${ROOT_DIR}/frontend"
  nohup setsid "$vite_bin" --host "$HOST" --port "$WEB_PORT" --strictPort \
    >"$WEB_LOG" 2>&1 &
  printf '%s\n' "$!" >"$WEB_PID_FILE"
  cd "$ROOT_DIR"
  if ! wait_for_service "$API_PID_FILE" api; then
    terminate_group "$WEB_PID_FILE" web
    cleanup_stale_pid "$API_PID_FILE" api
    tail -n 20 "$API_LOG" >&2 || true
    die "API failed to start on ${HOST}:${API_PORT}"
  fi
  if ! wait_for_service "$WEB_PID_FILE" web; then
    terminate_group "$API_PID_FILE" api
    cleanup_stale_pid "$WEB_PID_FILE" web
    tail -n 20 "$WEB_LOG" >&2 || true
    die "frontend failed to start on ${HOST}:${WEB_PORT}; Vite strict port mode will not select another port"
  fi
  status_dev
}

status_dev() {
  local pid
  cleanup_stale_pid "$API_PID_FILE" api
  cleanup_stale_pid "$WEB_PID_FILE" web
  printf 'api: '
  if managed_process "$API_PID_FILE" api; then
    pid="$(read_pid "$API_PID_FILE")"
    printf 'running (pid %s, pgid %s, sid %s, port %s)\n' "$pid" "$(process_group_id "$pid")" "$(process_session_id "$pid")" "$API_PORT"
  else
    printf 'stopped\n'
  fi
  printf 'web: '
  if managed_process "$WEB_PID_FILE" web; then
    pid="$(read_pid "$WEB_PID_FILE")"
    printf 'running (pid %s, pgid %s, sid %s, port %s)\n' "$pid" "$(process_group_id "$pid")" "$(process_session_id "$pid")" "$WEB_PORT"
  else
    printf 'stopped\n'
  fi
}

stop_dev() {
  terminate_group "$WEB_PID_FILE" web
  terminate_group "$API_PID_FILE" api
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
