"""
side_chains.jl — chi angle definitions, chiral center definitions, pseudo-beta atoms.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Chi angle atom definitions (dihedral angles for side chains)
# ──────────────────────────────────────────────────────────────────────────────

# For each residue: list of chi angles, each defined by 4 atom names.
const CHI_ANGLES_ATOMS = Dict{String, Vector{NTuple{4,String}}}(
    "ALA" => [],
    "ARG" => [
        ("N","CA","CB","CG"), ("CA","CB","CG","CD"),
        ("CB","CG","CD","NE"), ("CG","CD","NE","CZ"),
    ],
    "ASN" => [("N","CA","CB","CG"), ("CA","CB","CG","OD1")],
    "ASP" => [("N","CA","CB","CG"), ("CA","CB","CG","OD1")],
    "CYS" => [("N","CA","CB","SG")],
    "GLN" => [
        ("N","CA","CB","CG"), ("CA","CB","CG","CD"),
        ("CB","CG","CD","OE1"),
    ],
    "GLU" => [
        ("N","CA","CB","CG"), ("CA","CB","CG","CD"),
        ("CB","CG","CD","OE1"),
    ],
    "GLY" => [],
    "HIS" => [("N","CA","CB","CG"), ("CA","CB","CG","ND1")],
    "ILE" => [("N","CA","CB","CG1"), ("CA","CB","CG1","CD1")],
    "LEU" => [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "LYS" => [
        ("N","CA","CB","CG"), ("CA","CB","CG","CD"),
        ("CB","CG","CD","CE"), ("CG","CD","CE","NZ"),
    ],
    "MET" => [
        ("N","CA","CB","CG"), ("CA","CB","CG","SD"),
        ("CB","CG","SD","CE"),
    ],
    "PHE" => [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "PRO" => [("N","CA","CB","CG"), ("CA","CB","CG","CD")],
    "SER" => [("N","CA","CB","OG")],
    "THR" => [("N","CA","CB","OG1")],
    "TRP" => [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "TYR" => [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "VAL" => [("N","CA","CB","CG1")],
)

# ──────────────────────────────────────────────────────────────────────────────
# Chiral centers (for chirality checking)
# ──────────────────────────────────────────────────────────────────────────────

# Format: residue_name => Vector of (center_atom, neighbor1, neighbor2, neighbor3)
const CHIRAL_CENTERS = Dict{String, Vector{NTuple{4,String}}}(
    "ALA" => [("CA","N","C","CB")],
    "ARG" => [("CA","N","C","CB"), ("CZ","NE","NH1","NH2")],
    "ASN" => [("CA","N","C","CB")],
    "ASP" => [("CA","N","C","CB")],
    "CYS" => [("CA","N","C","CB")],
    "GLN" => [("CA","N","C","CB")],
    "GLU" => [("CA","N","C","CB")],
    "GLY" => [],
    "HIS" => [("CA","N","C","CB")],
    "ILE" => [("CA","N","C","CB"), ("CB","CA","CG1","CG2")],
    "LEU" => [("CA","N","C","CB")],
    "LYS" => [("CA","N","C","CB")],
    "MET" => [("CA","N","C","CB")],
    "PHE" => [("CA","N","C","CB")],
    "PRO" => [("CA","N","C","CB")],
    "SER" => [("CA","N","C","CB")],
    "THR" => [("CA","N","C","CB"), ("CB","CA","OG1","CG2")],
    "TRP" => [("CA","N","C","CB")],
    "TYR" => [("CA","N","C","CB")],
    "VAL" => [("CA","N","C","CB")],
)

"""
    get_chiral_centers(res_name::String) -> Vector{NTuple{4,String}}

Return chiral center definitions for a residue.
"""
function get_chiral_centers(res_name::String)::Vector{NTuple{4,String}}
    return get(CHIRAL_CENTERS, res_name, NTuple{4,String}[])
end

"""
    get_chi_angles(res_name::String) -> Vector{NTuple{4,String}}

Return chi angle atom definitions for a residue.
"""
function get_chi_angles(res_name::String)::Vector{NTuple{4,String}}
    return get(CHI_ANGLES_ATOMS, res_name, NTuple{4,String}[])
end

# ──────────────────────────────────────────────────────────────────────────────
# Pseudo-beta atom definitions
# ──────────────────────────────────────────────────────────────────────────────

# Atom used as pseudo-beta for each residue type.
# GLY uses CA; all others use CB; nucleic acids use C4'.
const PSEUDO_BETA_ATOM = Dict{String,String}(
    "GLY" => "CA",
)

"""
    get_pseudo_beta_atom(res_name::String, chain_type::String) -> String

Return the pseudo-beta atom name for a residue.
"""
function get_pseudo_beta_atom(res_name::String, chain_type::String)::String
    if chain_type == "polypeptide(L)"
        return get(PSEUDO_BETA_ATOM, res_name, "CB")
    elseif chain_type in ("polyribonucleotide", "polydeoxyribonucleotide")
        return "C4'"
    else
        return "CA"
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Rigid group atom indices for template featurization
# ──────────────────────────────────────────────────────────────────────────────

# Backbone frame atoms: (C, Cα, N) as 1-based indices into ATOM_ORDER
# Used to define the backbone rigid frame for each residue.
function _get_backbone_frame_indices()
    n_idx  = get_atom_index("N")
    ca_idx = get_atom_index("CA")
    c_idx  = get_atom_index("C")
    return (c_idx, ca_idx, n_idx)  # (C, Cα, N)
end

const BACKBONE_FRAME_INDICES = _get_backbone_frame_indices()

# RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX: (num_residue_types, num_rigid_groups, 3)
# For now we define the backbone group only (group 0) which uses N, Cα, C.
function _build_restype_rigidgroup_indices()
    n_restypes = length(ATOM_ORDER)
    # For each residue type, group 0 is backbone: indices of (C, Cα, N) in ATOM_ORDER
    idxs = Dict{String, Matrix{Int}}()
    for res in keys(_STANDARD_ATOMS_BY_RESIDUE)
        m = zeros(Int, 8, 3)
        n_i  = get_atom_index("N")
        ca_i = get_atom_index("CA")
        c_i  = get_atom_index("C")
        m[1, :] = [c_i, ca_i, n_i]  # backbone
        idxs[res] = m
    end
    return idxs
end

const RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX = _build_restype_rigidgroup_indices()
