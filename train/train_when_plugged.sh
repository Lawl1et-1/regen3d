#!/usr/bin/env bash
# Wait until the laptop runs on mains power, then train every organ (train_all.sh).
# Training a 3D NCA on battery throttles the GPU ~5x and would drain the battery.
# Run from the repo root:  bash train/train_when_plugged.sh liver kidney heart ...
online() {
  powershell.exe -NoProfile -Command "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus | Select-Object -First 1).PowerOnline" 2>/dev/null | tr -d '\r' | grep -qi true
}
until online; do sleep 60; done
echo "=== mains power detected $(date)"
ITERS=${ITERS:-1600} bash train/train_all.sh "$@"
