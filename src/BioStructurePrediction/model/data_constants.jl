"""
Constant arrays and lookup tables for model featurisation.
"""

# ──────────────────────────────────────────────
#  Residue type encoding
# ──────────────────────────────────────────────

# 32 residue types: 20 AA + 4 RNA (A,U,G,C) + 4 DNA (DA,DT,DG,DC) + UNK + gap + 2 reserved
const RESTYPE_ORDER = String[
    "ALA", "ARG", "ASN", "ASP", "CYS",   #  1-5
    "GLN", "GLU", "GLY", "HIS", "ILE",   #  6-10
    "LEU", "LYS", "MET", "PHE", "PRO",   # 11-15
    "SER", "THR", "TRP", "TYR", "VAL",   # 16-20
    "A",   "U",   "G",   "C",            # 21-24 (RNA)
    "DA",  "DT",  "DG",  "DC",           # 25-28 (DNA)
    "UNK", "GAP", "MSE", "SEC",          # 29-32
]
const NUM_RESIDUE_TYPES = 32

# Map residue name → 0-based type index
const RESTYPE_TO_IDX = Dict{String,Int}(
    name => (i - 1) for (i, name) in enumerate(RESTYPE_ORDER)
)

"""
    restype_index(res_name::String) -> Int

Return the 0-based residue type index (RESTYPE_TO_IDX).
Returns 28 (UNK) if not found.
"""
function restype_index(res_name::String)::Int
    return get(RESTYPE_TO_IDX, res_name, RESTYPE_TO_IDX["UNK"])
end

# ──────────────────────────────────────────────
#  Target feature encoding (447 dims)
# ──────────────────────────────────────────────

# The target_feat vector encodes:
#  0-31:   one-hot residue type (32 dims)
#  32-67:  one-hot element for each of 36 atom slots (36 dims total, see below)
#  68-...: various physicochemical features
#
# The full 447-dim encoding is defined empirically to match the reference model weights.
# The breakdown:
#   32 dims: residue type one-hot
#  128 dims: atom-type one-hot (128 element types)
#   39 dims: template pair features
#   64 dims: relative position encoding
#  184 dims: additional biochemical features (is_protein, is_rna, is_dna, is_ligand,
#             bond types, charges, masses, etc.)
#
# For simplicity this file defines the dimension constants only.
const TARGET_FEAT_DIM       = 447
const MSA_FEAT_DIM          = 34
const EXTRA_MSA_FEAT_DIM    = 25
const PAIR_FEAT_DIM         = 128
const SINGLE_FEAT_DIM       = 384
const TEMPLATE_FEAT_DIM     = 39
const RELATIVE_POS_DIM      = 139
const REF_ATOM_FEAT_DIM     = 128
const DIFFUSION_PAIR_DIM    = 16
const ATOM_PAIR_FEAT_DIM    = 16

# ──────────────────────────────────────────────
#  Confidence score bins
# ──────────────────────────────────────────────

# pLDDT: 50 bins uniformly in [0, 1]
const PLDDT_NUM_BINS  = 50
const PLDDT_BIN_EDGES = Float32.(LinRange(0f0, 1f0, PLDDT_NUM_BINS + 1))
const PLDDT_BIN_CENTERS = Float32.(0.5f0 .* (PLDDT_BIN_EDGES[1:end-1] .+ PLDDT_BIN_EDGES[2:end]))

# PAE: 64 bins from 0 to 32 Å
const PAE_NUM_BINS  = 64
const PAE_MAX_DIST  = 32f0
const PAE_BIN_EDGES = Float32.(LinRange(0f0, PAE_MAX_DIST, PAE_NUM_BINS + 1))
const PAE_BIN_CENTERS = Float32.(0.5f0 .* (PAE_BIN_EDGES[1:end-1] .+ PAE_BIN_EDGES[2:end]))

# Distogram: 64 bins from 2 to 22 Å (Cβ−Cβ distance)
const DISTOGRAM_NUM_BINS  = 64
const DISTOGRAM_MIN_DIST  = 2f0
const DISTOGRAM_MAX_DIST  = 22f0
const DISTOGRAM_BIN_EDGES = Float32.(LinRange(DISTOGRAM_MIN_DIST, DISTOGRAM_MAX_DIST, DISTOGRAM_NUM_BINS + 1))
const DISTOGRAM_BIN_CENTERS = Float32.(0.5f0 .* (DISTOGRAM_BIN_EDGES[1:end-1] .+ DISTOGRAM_BIN_EDGES[2:end]))

# ──────────────────────────────────────────────
#  Relative position encoding
# ──────────────────────────────────────────────

# Maximum relative sequence distance to encode (beyond this is clipped)
const MAX_RELATIVE_IDX = 65
# Relative position bins: -MAX_RELATIVE_IDX, ..., -1, 0, 1, ..., MAX_RELATIVE_IDX, same_chain
const NUM_RELATIVE_POS_BINS = 2 * MAX_RELATIVE_IDX + 2  # = 132 (+1 for same-chain bin) → 139 total

# ──────────────────────────────────────────────
#  Atom type indices for confidence head
# ──────────────────────────────────────────────

# 24 atom types used in pLDDT/experimentally_resolved heads
const ATOM_TYPE_ORDER_PLDDT = String[
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2",
    "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1", "ND2",
    "OD1", "OD2", "SD", "CE", "CE1", "CE2", "NE", "NE1",
]
const NUM_ATOM_TYPES_PLDDT = 24

const ATOM_TYPE_TO_PLDDT_IDX = Dict{String,Int}(
    a => (i-1) for (i, a) in enumerate(ATOM_TYPE_ORDER_PLDDT)
)
