#!/usr/bin/env bash
set -euo pipefail

systemctl --user disable --now gpu-slack-runner.timer || true
rm -f "$HOME/.config/systemd/user/gpu-slack-runner.service"
rm -f "$HOME/.config/systemd/user/gpu-slack-runner.timer"
systemctl --user daemon-reload

echo "Uninstalled user systemd timer."
