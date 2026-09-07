#!/usr/bin/env bash
set -euxo pipefail

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi

docker pull "${docker_image}"
docker stop "${container_name}" 2>/dev/null || true
docker rm   "${container_name}" 2>/dev/null || true
docker run -d --name "${container_name}" --restart unless-stopped \
  -p "${app_port}:${app_port}" -e MODEL_PATH="${model_path}" "${docker_image}"

echo "startup-script: ${container_name} is up on port ${app_port}"
