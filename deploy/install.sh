#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/Luqmaan-Belim/invoice-renamer.git"
APP_DIR="/opt/invoice-renamer"
ENV_FILE="/etc/invoice-renamer.env"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer as root (sudo)." >&2
  exit 1
fi

apt-get update
apt-get install -y git python3 python3-venv tesseract-ocr libgl1 libglib2.0-0

if [[ -d "${APP_DIR}/.git" ]]; then
  git -C "${APP_DIR}" fetch origin main
  git -C "${APP_DIR}" reset --hard origin/main
else
  rm -rf "${APP_DIR}"
  git clone --branch main "${REPO_URL}" "${APP_DIR}"
fi

python3 -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/pip" install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

chown -R ubuntu:ubuntu "${APP_DIR}"

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${APP_DIR}/deploy/invoice-renamer.env.example" "${ENV_FILE}"
  chmod 600 "${ENV_FILE}"
  echo "Created ${ENV_FILE}. Fill in the real credentials, then rerun this script." >&2
  exit 2
fi

install -m 0644 "${APP_DIR}/deploy/invoice-renamer.service" /etc/systemd/system/invoice-renamer.service
install -m 0644 "${APP_DIR}/deploy/invoice-renamer.timer" /etc/systemd/system/invoice-renamer.timer

systemctl daemon-reload
systemctl enable --now invoice-renamer.timer
systemctl start invoice-renamer.service

systemctl --no-pager --full status invoice-renamer.timer
systemctl --no-pager --full status invoice-renamer.service || true
