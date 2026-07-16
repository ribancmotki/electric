#!/usr/bin/env bash
# gcp_mount_ssd.sh — Prepare and mount a local NVMe SSD on a Google Cloud
# instance (e.g. n1-standard-8 with local SSD attached).
#
# Usage:
#   sudo ./scripts/gcp_mount_ssd.sh [--device /dev/nvme0n1] [--mount_dir /nvme]
#
# This script:
#   1. Detects the first available local SSD device (or uses --device).
#   2. Formats it as ext4 (if not already formatted).
#   3. Mounts it at --mount_dir.
#   4. Sets permissions for the current user.
#
# Run as root or with sudo.

set -euo pipefail

DEVICE="${DEVICE:-}"
MOUNT_DIR="${MOUNT_DIR:-/nvme}"
FSTYPE="${FSTYPE:-ext4}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)    DEVICE="$2"; shift 2 ;;
        --mount_dir) MOUNT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: sudo $0 [--device /dev/nvme0n1] [--mount_dir /nvme]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "[INFO] Mount directory: $MOUNT_DIR"

# ── Auto-detect NVMe device if not specified ───────────────────────────────────
if [[ -z "$DEVICE" ]]; then
    # Look for unmounted NVMe or local SSD devices
    DEVICE=$(lsblk -nd -o NAME,TYPE | awk '$2=="disk" {print "/dev/"$1}' | \
             while IFS= read -r dev; do
                 mount | grep -q "^$dev" || echo "$dev"
             done | head -1)
fi

if [[ -z "$DEVICE" ]]; then
    echo "[ERROR] No available block device found. Use --device to specify one."
    exit 1
fi

echo "[INFO] Using device: $DEVICE"

# ── Check if already formatted ────────────────────────────────────────────────
EXISTING_FSTYPE=$(blkid -s TYPE -o value "$DEVICE" 2>/dev/null || echo "")

if [[ -z "$EXISTING_FSTYPE" ]]; then
    echo "[INFO] Formatting $DEVICE as $FSTYPE ..."
    mkfs -t "$FSTYPE" -F "$DEVICE"
else
    echo "[INFO] Device already formatted as $EXISTING_FSTYPE — skipping format."
fi

# ── Mount ────────────────────────────────────────────────────────────────────
if mountpoint -q "$MOUNT_DIR"; then
    echo "[INFO] $MOUNT_DIR is already mounted."
else
    mkdir -p "$MOUNT_DIR"
    echo "[INFO] Mounting $DEVICE → $MOUNT_DIR ..."
    mount -o discard,defaults "$DEVICE" "$MOUNT_DIR"
fi

# ── Permissions ──────────────────────────────────────────────────────────────
CURRENT_USER="${SUDO_USER:-$(whoami)}"
chown -R "$CURRENT_USER:$CURRENT_USER" "$MOUNT_DIR"
chmod 755 "$MOUNT_DIR"

# ── Performance tuning ────────────────────────────────────────────────────────
# Enable write-back caching for faster sequential writes
if [[ -f "/sys/block/$(basename "$DEVICE")/queue/write_cache" ]]; then
    echo "write back" > "/sys/block/$(basename "$DEVICE")/queue/write_cache" || true
fi

# Increase read-ahead for large sequential DB files
blockdev --setra 16384 "$DEVICE" || true

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[INFO] SSD mounted successfully."
echo "[INFO] Device:      $DEVICE"
echo "[INFO] Mount point: $MOUNT_DIR"
df -h "$MOUNT_DIR"
echo ""
echo "[INFO] Next steps:"
echo "  1. Copy databases: ./scripts/copy_to_ssd.sh --ssd_dir $MOUNT_DIR/databases"
echo "  2. Run predictions: DB_DIR=$MOUNT_DIR/databases julia run_prediction.jl ..."
