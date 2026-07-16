"""
data_constants.jl — Model architecture constants, bin edges, residue type ordering.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Residue type ordering (32 types including gap and unknowns)
# ──────────────────────────────────────────────────────────────────────────────

const RESTYPE_ORDER = POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP

# ──────────────────────────────────────────────────────────────────────────────
# pLDDT bins
# ──────────────────────────────────────────────────────────────────────────────

const PLDDT_NUM_BINS = 50
const PLDDT_BIN_CENTERS = Float32.(range(0.5/PLDDT_NUM_BINS, 1.0 - 0.5/PLDDT_NUM_BINS,
                                          length=PLDDT_NUM_BINS))

# ──────────────────────────────────────────────────────────────────────────────
# PAE bins (64 bins, 0–31 Å)
# ──────────────────────────────────────────────────────────────────────────────

const PAE_MAX_ERROR_BIN = 31.0f0
const PAE_NUM_BINS      = 64

function _make_pae_bin_centers()::Vector{Float32}
    edges = collect(Float32, range(0, PAE_MAX_ERROR_BIN, length=PAE_NUM_BINS-1))
    step  = edges[2] - edges[1]
    centers = [e + step/2 for e in edges]
    push!(centers, PAE_MAX_ERROR_BIN + step)  # catch-all bin
    return centers
end

const PAE_BIN_CENTERS = _make_pae_bin_centers()

# ──────────────────────────────────────────────────────────────────────────────
# Distogram bins (64 bins, 2.3125–21.6875 Å)
# ──────────────────────────────────────────────────────────────────────────────

const DISTOGRAM_FIRST_BREAK = 2.3125f0
const DISTOGRAM_LAST_BREAK  = 21.6875f0
const DISTOGRAM_NUM_BINS    = 64

const DISTOGRAM_BIN_EDGES = Float32.(range(DISTOGRAM_FIRST_BREAK, DISTOGRAM_LAST_BREAK,
                                            length=DISTOGRAM_NUM_BINS-1))

# ──────────────────────────────────────────────────────────────────────────────
# PDE bins (64 bins, 0–31 Å)
# ──────────────────────────────────────────────────────────────────────────────

const PDE_MAX_ERROR_BIN = 31.0f0
const PDE_NUM_BINS      = 64

# ──────────────────────────────────────────────────────────────────────────────
# Feature dimensions
# ──────────────────────────────────────────────────────────────────────────────

const TARGET_FEAT_DIM = POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 4  # aatype_OH + 4 flags
const MSA_FEAT_DIM    = POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 3  # aatype_OH + del_feats

# ──────────────────────────────────────────────────────────────────────────────
# Relative position constants
# ──────────────────────────────────────────────────────────────────────────────

const MAX_RELATIVE_IDX   = 32
const MAX_RELATIVE_CHAIN = 2
const REL_POS_NUM_BINS   = 2 * MAX_RELATIVE_IDX + 2  # 66

# ──────────────────────────────────────────────────────────────────────────────
# Architecture dimensions
# ──────────────────────────────────────────────────────────────────────────────

const C_S    = 384   # single/sequence channel
const C_Z    = 128   # pair channel
const C_MSA  = 64    # MSA channel
const C_ATOM = 128   # per-atom channel

# Bucket sizes for padding
const DEFAULT_BUCKETS = Int[
    256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120
]

"""
    select_bucket(n::Int, buckets::Vector{Int}) -> Int

Select the smallest bucket size that is ≥ n. Returns n if no bucket fits.
"""
function select_bucket(n::Int, buckets::Vector{Int})::Int
    for b in sort(buckets)
        b >= n && return b
    end
    return n  # no bucket large enough; use exact size
end

# ──────────────────────────────────────────────────────────────────────────────
# Atom type ordering for pLDDT (37 slots)
# ──────────────────────────────────────────────────────────────────────────────

const ATOM_TYPE_ORDER_PLDDT = ATOM_ORDER
