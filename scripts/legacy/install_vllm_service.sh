#!/bin/bash
# Legacy Qwen vLLM service installer.
# Usage: bash /mnt/c/Users/ASUS/Desktop/echo/scripts/legacy/install_vllm_service.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SRC="$SCRIPT_DIR/vllm.service"
SERVICE_DST="/etc/systemd/system/vllm.service"

echo "Installing vLLM systemd service..."
sudo cp "$SERVICE_SRC" "$SERVICE_DST"
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm

echo "Done. Check status with: sudo systemctl status vllm"
echo "View logs with: sudo journalctl -u vllm -f"
