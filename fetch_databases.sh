#!/usr/bin/env bash
# fetch_databases.sh — Download all sequence and template databases required for inference.
#
# Usage:
#   ./fetch_databases.sh [--db_dir /path/to/databases]
#
# Environment variables:
#   DB_DIR      — target directory (default: ./databases)
#   MAX_JOBS    — max parallel downloads (default: 4)
#   RETRY       — number of retry attempts (default: 3)
#
# Databases downloaded:
#   1. UniRef90  (UniProt 2022-05)
#   2. MGnify    (2022-05)
#   3. UniProt   (cluster annotations, 2021-04)
#   4. Small BFD (first-non-consensus sequences)
#   5. NT (nTrRNA 2023-02-23)
#   6. Rfam      (14.4)
#   7. RNAcentral (active sequences, 90% id, 80% cov)
#   8. PDB mmCIFs (2022-09-28)
#   9. PDB seqres (2022-09-28)

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────

DB_DIR="${DB_DIR:-$(pwd)/databases}"
MAX_JOBS="${MAX_JOBS:-4}"
RETRY="${RETRY:-3}"

# Parse optional --db_dir flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --db_dir)   DB_DIR="$2"; shift 2 ;;
        --max_jobs) MAX_JOBS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--db_dir DIR] [--max_jobs N]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Database directory: $DB_DIR"
mkdir -p "$DB_DIR"

# ── Utility functions ──────────────────────────────────────────────────────────

RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GRN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YEL}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

download() {
    local url="$1"
    local dest="$2"
    local desc="${3:-$dest}"

    if [[ -f "$dest" ]]; then
        log_info "Already present, skipping: $desc"
        return 0
    fi

    log_info "Downloading: $desc"
    local attempt=0
    while (( attempt < RETRY )); do
        if wget -q --show-progress --retry-connrefused --waitretry=10 \
               --read-timeout=120 -O "${dest}.tmp" "$url"; then
            mv "${dest}.tmp" "$dest"
            log_info "Done: $desc"
            return 0
        fi
        attempt=$((attempt + 1))
        log_warn "Attempt $attempt/$RETRY failed for $desc; retrying..."
        sleep 5
    done
    log_error "Failed after $RETRY attempts: $desc"
    rm -f "${dest}.tmp"
    return 1
}

download_zst_unpack() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"
    local desc="${4:-$filename}"
    local zst_path="$dest_dir/$filename.zst"
    local out_path="$dest_dir/$filename"

    mkdir -p "$dest_dir"

    if [[ -f "$out_path" ]]; then
        log_info "Already present, skipping: $desc"
        return 0
    fi

    if [[ ! -f "$zst_path" ]]; then
        download "$url" "$zst_path" "$desc (compressed)"
    fi

    log_info "Decompressing: $desc"
    if command -v zstd &>/dev/null; then
        zstd -d "$zst_path" -o "$out_path"
    else
        log_error "zstd not found; cannot decompress $zst_path"
        return 1
    fi
    rm -f "$zst_path"
}

# ── Job control ────────────────────────────────────────────────────────────────

declare -a PIDS=()
FAILED=0

wait_for_slot() {
    while (( ${#PIDS[@]} >= MAX_JOBS )); do
        # Wait for any child to finish
        local new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            else
                wait "$pid" || FAILED=$((FAILED+1))
            fi
        done
        PIDS=("${new_pids[@]}")
        (( ${#PIDS[@]} < MAX_JOBS )) && break
        sleep 1
    done
}

run_bg() {
    wait_for_slot
    "$@" &
    PIDS+=("$!")
}

wait_all() {
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAILED=$((FAILED+1))
    done
    PIDS=()
}

# ── Database URLs ──────────────────────────────────────────────────────────────

UNIREF90_URL="https://storage.googleapis.com/alphafold-databases/v0/databases/uniref90_2022_05.fasta.zst"
MGNIFY_URL="https://storage.googleapis.com/alphafold-databases/v0/databases/mgy_clusters_2022_05.fasta.zst"
UNIPROT_URL="https://storage.googleapis.com/alphafold-databases/v0/databases/uniprot_all_2021_04.fasta.zst"
SMALL_BFD_URL="https://storage.googleapis.com/alphafold-databases/v0/databases/bfd-first_non_consensus_sequences.fasta.zst"
NTRNA_URL="https://storage.googleapis.com/alphafold-databases/v0/databases/nt_all_2023_02_23.fasta.zst"
RFAM_URL="https://storage.googleapis.com/alphafold-databases/v0/databases/rfam_14_4_clustered_rep_seq.fasta.zst"
RNACENTRAL_URL="https://storage.googleapis.com/alphafold-databases/v0/databases/rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst"
PDB_SEQRES_URL="https://storage.googleapis.com/alphafold-databases/v0/databases/pdb_seqres_2022_09_28.fasta.zst"

# PDB mmCIFs — large tar archive (sharded)
PDB_MMCIF_BASE="https://storage.googleapis.com/alphafold-databases/v0/databases"
PDB_MMCIF_TAR="pdb_2022_09_28_mmcifs.tar"

# ── Download each database in background ──────────────────────────────────────

log_info "Starting database downloads (max $MAX_JOBS parallel jobs) ..."

# 1. UniRef90
run_bg bash -c "
    mkdir -p '$DB_DIR/uniref90'
    if [[ ! -f '$DB_DIR/uniref90/uniref90_2022_05.fasta' ]]; then
        download '$UNIREF90_URL' '$DB_DIR/uniref90/uniref90_2022_05.fasta.zst' 'UniRef90'
        zstd -d '$DB_DIR/uniref90/uniref90_2022_05.fasta.zst' -o '$DB_DIR/uniref90/uniref90_2022_05.fasta'
        rm -f '$DB_DIR/uniref90/uniref90_2022_05.fasta.zst'
    fi
"

# 2. MGnify
run_bg bash -c "
    mkdir -p '$DB_DIR/mgnify'
    if [[ ! -f '$DB_DIR/mgnify/mgy_clusters_2022_05.fa' ]]; then
        download '$MGNIFY_URL' '$DB_DIR/mgnify/mgy_clusters_2022_05.fa.zst' 'MGnify'
        zstd -d '$DB_DIR/mgnify/mgy_clusters_2022_05.fa.zst' -o '$DB_DIR/mgnify/mgy_clusters_2022_05.fa'
        rm -f '$DB_DIR/mgnify/mgy_clusters_2022_05.fa.zst'
    fi
"

# 3. UniProt cluster annotations
run_bg bash -c "
    mkdir -p '$DB_DIR/uniprot'
    if [[ ! -f '$DB_DIR/uniprot/uniprot_all_2021_04.fa' ]]; then
        download '$UNIPROT_URL' '$DB_DIR/uniprot/uniprot_all_2021_04.fa.zst' 'UniProt'
        zstd -d '$DB_DIR/uniprot/uniprot_all_2021_04.fa.zst' -o '$DB_DIR/uniprot/uniprot_all_2021_04.fa'
        rm -f '$DB_DIR/uniprot/uniprot_all_2021_04.fa.zst'
    fi
"

# 4. Small BFD
run_bg bash -c "
    mkdir -p '$DB_DIR/small_bfd'
    if [[ ! -f '$DB_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta' ]]; then
        download '$SMALL_BFD_URL' '$DB_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta.zst' 'Small BFD'
        zstd -d '$DB_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta.zst' -o '$DB_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta'
        rm -f '$DB_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta.zst'
    fi
"

# 5. NT RNA
run_bg bash -c "
    mkdir -p '$DB_DIR/nt_rna'
    if [[ ! -f '$DB_DIR/nt_rna/nt_all_2023_02_23.fasta' ]]; then
        download '$NTRNA_URL' '$DB_DIR/nt_rna/nt_all_2023_02_23.fasta.zst' 'NT RNA'
        zstd -d '$DB_DIR/nt_rna/nt_all_2023_02_23.fasta.zst' -o '$DB_DIR/nt_rna/nt_all_2023_02_23.fasta'
        rm -f '$DB_DIR/nt_rna/nt_all_2023_02_23.fasta.zst'
    fi
"

# 6. Rfam
run_bg bash -c "
    mkdir -p '$DB_DIR/rfam'
    if [[ ! -f '$DB_DIR/rfam/rfam_14_4_clustered_rep_seq.fasta' ]]; then
        download '$RFAM_URL' '$DB_DIR/rfam/rfam_14_4_clustered_rep_seq.fasta.zst' 'Rfam'
        zstd -d '$DB_DIR/rfam/rfam_14_4_clustered_rep_seq.fasta.zst' -o '$DB_DIR/rfam/rfam_14_4_clustered_rep_seq.fasta'
        rm -f '$DB_DIR/rfam/rfam_14_4_clustered_rep_seq.fasta.zst'
    fi
"

# 7. RNAcentral
run_bg bash -c "
    mkdir -p '$DB_DIR/rnacentral'
    if [[ ! -f '$DB_DIR/rnacentral/rnacentral_active_seq_id_90_cov_80_linclust.fasta' ]]; then
        download '$RNACENTRAL_URL' \
            '$DB_DIR/rnacentral/rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst' 'RNAcentral'
        zstd -d '$DB_DIR/rnacentral/rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst' \
             -o '$DB_DIR/rnacentral/rnacentral_active_seq_id_90_cov_80_linclust.fasta'
        rm -f '$DB_DIR/rnacentral/rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst'
    fi
"

# 8. PDB seqres
run_bg bash -c "
    mkdir -p '$DB_DIR/seqres'
    if [[ ! -f '$DB_DIR/seqres/pdb_seqres_2022_09_28.fasta' ]]; then
        download '$PDB_SEQRES_URL' '$DB_DIR/seqres/pdb_seqres_2022_09_28.fasta.zst' 'PDB seqres'
        zstd -d '$DB_DIR/seqres/pdb_seqres_2022_09_28.fasta.zst' \
             -o '$DB_DIR/seqres/pdb_seqres_2022_09_28.fasta'
        rm -f '$DB_DIR/seqres/pdb_seqres_2022_09_28.fasta.zst'
    fi
"

# 9. PDB mmCIFs (sequential: large tar, extract in place)
run_bg bash -c "
    mkdir -p '$DB_DIR/pdb'
    if [[ ! -f '$DB_DIR/pdb/$PDB_MMCIF_TAR' && ! -d '$DB_DIR/pdb/mmcifs' ]]; then
        log_info 'Downloading PDB mmCIFs ...'
        download '$PDB_MMCIF_BASE/$PDB_MMCIF_TAR' '$DB_DIR/pdb/$PDB_MMCIF_TAR' 'PDB mmCIFs'
    fi
    if [[ -f '$DB_DIR/pdb/$PDB_MMCIF_TAR' && ! -d '$DB_DIR/pdb/mmcifs' ]]; then
        log_info 'Extracting PDB mmCIFs ...'
        tar -xf '$DB_DIR/pdb/$PDB_MMCIF_TAR' -C '$DB_DIR/pdb/'
        log_info 'PDB mmCIFs extracted.'
    fi
"

# Wait for all downloads
wait_all

if (( FAILED > 0 )); then
    log_error "$FAILED download(s) failed. Check the output above."
    exit 1
fi

log_info "All databases downloaded to $DB_DIR"
log_info "Total size: $(du -sh '$DB_DIR' | awk '{print $1}')"
