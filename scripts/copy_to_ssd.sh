#!/usr/bin/env bash
# copy_to_ssd.sh — Copy databases from a slow mount (e.g. HDD/NFS) to a fast
# NVMe SSD for low-latency jackhmmer/nhmmer search.
#
# Usage:
#   ./scripts/copy_to_ssd.sh [--src_dir /data/databases] [--ssd_dir /nvme/databases]
#
# The script preserves directory structure and skips already-copied files.

set -euo pipefail

SRC_DIR="${SRC_DIR:-${DB_DIR:-$(pwd)/databases}}"
SSD_DIR="${SSD_DIR:-/nvme/databases}"
MAX_JOBS="${MAX_JOBS:-4}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src_dir)  SRC_DIR="$2"; shift 2 ;;
        --ssd_dir)  SSD_DIR="$2"; shift 2 ;;
        --max_jobs) MAX_JOBS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--src_dir DIR] [--ssd_dir DIR] [--max_jobs N]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "[INFO] Copying databases from: $SRC_DIR"
echo "[INFO] Destination SSD: $SSD_DIR"

# Check available space on SSD
SSD_AVAIL_GB=$(df -BG "$SSD_DIR" | awk 'NR==2 {gsub("G",""); print $4}')
SRC_SIZE_GB=$(du -sBG "$SRC_DIR" 2>/dev/null | awk '{gsub("G",""); print $1}' || echo 0)
echo "[INFO] SSD available: ${SSD_AVAIL_GB} GiB  |  Source size: ${SRC_SIZE_GB} GiB"

if (( SRC_SIZE_GB > SSD_AVAIL_GB )); then
    echo "[WARN] Source size (${SRC_SIZE_GB} GiB) exceeds SSD free space (${SSD_AVAIL_GB} GiB)."
    echo "[WARN] Copying anyway — you may need to reduce database selection."
fi

mkdir -p "$SSD_DIR"

# Use rsync for efficient incremental copy
rsync -aH \
    --info=progress2 \
    --no-inc-recursive \
    --partial \
    --partial-dir=".rsync-partial" \
    --exclude="*.tmp" \
    --exclude=".rsync-partial" \
    "$SRC_DIR/" "$SSD_DIR/"

echo "[INFO] Copy complete."
echo "[INFO] SSD databases directory: $SSD_DIR"
echo "[INFO] Export DB_DIR=$SSD_DIR before running predictions for faster search."
