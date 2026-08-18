#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-agent-workbench}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-agent-workbench}"
PORT="${PORT:-8501}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-90}"
BUILD_IMAGE=true
PULL_IMAGE=false
PUSH_IMAGE=false

usage() {
    cat <<'EOF'
Build and deploy the Agent Workbench as a Docker container.

Usage: ./scripts/deploy.sh [options]

Options:
  --image NAME       Image repository/name (default: agent-workbench)
  --tag TAG          Image tag (default: latest)
  --container NAME   Container name (default: agent-workbench)
  --port PORT        Host port mapped to container port 8501 (default: 8501)
  --env-file PATH    Runtime environment file (default: .env)
  --health-timeout S Seconds to wait for a healthy container (default: 90)
  --no-build         Run an existing local image without building
  --pull             Pull the image before deployment; implies --no-build
  --push             Push the image after a successful build
  -h, --help         Show this help

Examples:
  ./scripts/deploy.sh
  PORT=8080 ./scripts/deploy.sh
  ./scripts/deploy.sh --image ghcr.io/owner/agent --tag latest --pull
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --image) IMAGE_NAME="${2:?--image requires a value}"; shift 2 ;;
        --tag) IMAGE_TAG="${2:?--tag requires a value}"; shift 2 ;;
        --container) CONTAINER_NAME="${2:?--container requires a value}"; shift 2 ;;
        --port) PORT="${2:?--port requires a value}"; shift 2 ;;
        --env-file) ENV_FILE="${2:?--env-file requires a value}"; shift 2 ;;
        --health-timeout) HEALTH_TIMEOUT="${2:?--health-timeout requires a value}"; shift 2 ;;
        --no-build) BUILD_IMAGE=false; shift ;;
        --pull) PULL_IMAGE=true; BUILD_IMAGE=false; shift ;;
        --push) PUSH_IMAGE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"
[[ "${PORT}" =~ ^[0-9]+$ ]] || die "port must be numeric: ${PORT}"
((PORT >= 1 && PORT <= 65535)) || die "port must be between 1 and 65535"
[[ "${HEALTH_TIMEOUT}" =~ ^[0-9]+$ ]] || die "health timeout must be numeric"

IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"

if [[ "${BUILD_IMAGE}" == true ]]; then
    printf 'Building %s...\n' "${IMAGE_REF}"
    docker build --pull --tag "${IMAGE_REF}" "${PROJECT_ROOT}"
elif [[ "${PULL_IMAGE}" == true ]]; then
    printf 'Pulling %s...\n' "${IMAGE_REF}"
    docker pull "${IMAGE_REF}"
else
    docker image inspect "${IMAGE_REF}" >/dev/null 2>&1 \
        || die "image does not exist locally: ${IMAGE_REF}"
fi

if [[ "${PUSH_IMAGE}" == true ]]; then
    [[ "${BUILD_IMAGE}" == true ]] || die "--push requires an image build"
    printf 'Pushing %s...\n' "${IMAGE_REF}"
    docker push "${IMAGE_REF}"
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    printf 'Replacing container %s...\n' "${CONTAINER_NAME}"
    docker rm --force "${CONTAINER_NAME}" >/dev/null
fi

run_args=(
    run --detach
    --name "${CONTAINER_NAME}"
    --restart unless-stopped
    --publish "${PORT}:8501"
)

if [[ -f "${ENV_FILE}" ]]; then
    run_args+=(--env-file "${ENV_FILE}")
else
    printf 'warning: environment file not found: %s\n' "${ENV_FILE}" >&2
fi

run_args+=("${IMAGE_REF}")
container_id="$(docker "${run_args[@]}")"

printf 'Waiting for %s to become healthy...\n' "${CONTAINER_NAME}"
deadline=$((SECONDS + HEALTH_TIMEOUT))
while ((SECONDS < deadline)); do
    health="$(
        docker container inspect \
            --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
            "${CONTAINER_NAME}" 2>/dev/null || true
    )"
    case "${health}" in
        healthy) break ;;
        unhealthy|exited|dead)
            docker logs --tail 100 "${CONTAINER_NAME}" >&2 || true
            die "container entered ${health} state"
            ;;
    esac
    sleep 2
done

[[ "${health:-}" == "healthy" ]] || {
    docker logs --tail 100 "${CONTAINER_NAME}" >&2 || true
    die "container did not become healthy within ${HEALTH_TIMEOUT} seconds"
}

printf 'Deployed %s (%s) at http://localhost:%s\n' \
    "${CONTAINER_NAME}" "${container_id:0:12}" "${PORT}"
