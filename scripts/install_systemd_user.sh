#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_PATH="$(pwd)"
CONFIG_PATH="${1:-$REPO_PATH/configs/default.yaml}"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"

sed \
  -e "s#__REPO_PATH__#$REPO_PATH#g" \
  -e "s#__CONFIG_PATH__#$CONFIG_PATH#g" \
  systemd/gpu-slack-runner.service.template > "$SYSTEMD_USER_DIR/gpu-slack-runner.service"

cp systemd/gpu-slack-runner.timer.template "$SYSTEMD_USER_DIR/gpu-slack-runner.timer"

systemctl --user daemon-reload
systemctl --user enable --now gpu-slack-runner.timer

echo "Installed user systemd timer."
echo "Status: systemctl --user status gpu-slack-runner.timer"
echo "Logs:   journalctl --user -u gpu-slack-runner.service -f"
