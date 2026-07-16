"""
atom_types.jl — atom name constants, protonation state definitions, and VDW radii.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard atom ordering (37 heavy-atom slots per residue)
# ──────────────────────────────────────────────────────────────────────────────

const ATOM_ORDER = (
    "N",   "CA",  "C",   "CB",  "O",   "CG",  "CG1", "CG2", "OG",
    "OG1", "SG",  "CD",  "CD1", "CD2", "ND1", "ND2", "OD1", "OD2",
    "SD",  "CE",  "CE1", "CE2", "CE3", "NE",  "NE1", "NE2", "OE1",
    "OE2", "CH2", "NH1", "NH2", "OH",  "CZ",  "CZ2", "CZ3", "NZ",
    "OXT",
)

const NUM_ATOM_SLOTS = length(ATOM_ORDER)  # 37

const ATOM_ORDER_INDEX = Dict{String,Int}(
    name => i for (i, name) in enumerate(ATOM_ORDER)
)

"""
    get_atom_index(atom_name::String) -> Int

Return 1-based index of atom_name in ATOM_ORDER, or 0 if not found.
"""
function get_atom_index(atom_name::String)::Int
    return get(ATOM_ORDER_INDEX, atom_name, 0)
end

# ──────────────────────────────────────────────────────────────────────────────
# Per-residue standard atom mask (which of the 37 slots are valid)
# ──────────────────────────────────────────────────────────────────────────────

const _STANDARD_ATOMS_BY_RESIDUE = Dict{String, Vector{String}}(
    "ALA" => ["N","CA","C","O","CB"],
    "ARG" => ["N","CA","C","O","CB","CG","CD","NE","CZ","NH1","NH2"],
    "ASN" => ["N","CA","C","O","CB","CG","OD1","ND2"],
    "ASP" => ["N","CA","C","O","CB","CG","OD1","OD2"],
    "CYS" => ["N","CA","C","O","CB","SG"],
    "GLN" => ["N","CA","C","O","CB","CG","CD","OE1","NE2"],
    "GLU" => ["N","CA","C","O","CB","CG","CD","OE1","OE2"],
    "GLY" => ["N","CA","C","O"],
    "HIS" => ["N","CA","C","O","CB","CG","ND1","CD2","CE1","NE2"],
    "ILE" => ["N","CA","C","O","CB","CG1","CG2","CD1"],
    "LEU" => ["N","CA","C","O","CB","CG","CD1","CD2"],
    "LYS" => ["N","CA","C","O","CB","CG","CD","CE","NZ"],
    "MET" => ["N","CA","C","O","CB","CG","SD","CE"],
    "PHE" => ["N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ"],
    "PRO" => ["N","CA","C","O","CB","CG","CD"],
    "SER" => ["N","CA","C","O","CB","OG"],
    "THR" => ["N","CA","C","O","CB","OG1","CG2"],
    "TRP" => ["N","CA","C","O","CB","CG","CD1","CD2","CE2","CE3","NE1","CZ2","CZ3","CH2"],
    "TYR" => ["N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ","OH"],
    "VAL" => ["N","CA","C","O","CB","CG1","CG2"],
    "UNK" => ["N","CA","C","O"],
    "MSE" => ["N","CA","C","O","CB","CG","SD","CE"],
)

function _build_standard_atom_mask()::Dict{String, BitVector}
    result = Dict{String, BitVector}()
    for (res, atoms) in _STANDARD_ATOMS_BY_RESIDUE
        mask = falses(NUM_ATOM_SLOTS)
        for a in atoms
            idx = get_atom_index(a)
            idx > 0 && (mask[idx] = true)
        end
        result[res] = mask
    end
    return result
end

const STANDARD_ATOM_MASK = _build_standard_atom_mask()

"""
    get_standard_atoms(res_name::String) -> Vector{String}

Return the list of standard heavy-atom names for a residue.
"""
function get_standard_atoms(res_name::String)::Vector{String}
    return get(_STANDARD_ATOMS_BY_RESIDUE, res_name, String[])
end

# ──────────────────────────────────────────────────────────────────────────────
# Protonation state: hydrogen atoms that may be deprotonated
# ──────────────────────────────────────────────────────────────────────────────

const PROTONATION_HYDROGENS = Dict{String,Set{String}}(
    "ASP" => Set(["HD2"]),
    "GLU" => Set(["HE2"]),
    "HIS" => Set(["HD1","HE2"]),
    "CYS" => Set(["HG"]),
    "TYR" => Set(["HH"]),
    "LYS" => Set(["HZ1","HZ2","HZ3"]),
    "ARG" => Set(["HH11","HH12","HH21","HH22","HE"]),
    "SER" => Set(["HG"]),
    "THR" => Set(["HG1"]),
)

# ──────────────────────────────────────────────────────────────────────────────
# Van der Waals radii (Angstroms) by element
# ──────────────────────────────────────────────────────────────────────────────

const VDW_RADII = Dict{String,Float32}(
    "H"  => 1.20f0,
    "C"  => 1.70f0,
    "N"  => 1.55f0,
    "O"  => 1.52f0,
    "F"  => 1.47f0,
    "P"  => 1.80f0,
    "S"  => 1.80f0,
    "Cl" => 1.75f0,
    "Br" => 1.85f0,
    "I"  => 1.98f0,
    "Se" => 1.90f0,
    "B"  => 1.92f0,
    "Si" => 2.10f0,
    "Fe" => 2.05f0,
    "Zn" => 2.10f0,
    "Cu" => 2.00f0,
    "Mn" => 2.05f0,
    "Ca" => 2.31f0,
    "Mg" => 1.73f0,
    "Na" => 2.27f0,
    "K"  => 2.75f0,
)

const DEFAULT_VDW_RADIUS = 1.80f0

"""
    get_vdw_radius(element::String) -> Float32

Return the van der Waals radius for an element. Falls back to DEFAULT_VDW_RADIUS.
"""
function get_vdw_radius(element::String)::Float32
    return get(VDW_RADII, element, DEFAULT_VDW_RADIUS)
end

# ──────────────────────────────────────────────────────────────────────────────
# Atom name character encoding helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    atom_name_to_chars(atom_name::String) -> NTuple{4,Int32}

Encode atom name as up to 4 ASCII code values minus 32 (for network embedding).
Pads with zeros if shorter than 4 characters.
"""
function atom_name_to_chars(atom_name::String)::NTuple{4,Int32}
    chars = Int32[0, 0, 0, 0]
    for (i, ch) in enumerate(atom_name)
        i > 4 && break
        chars[i] = Int32(codepoint(ch)) - Int32(32)
    end
    return (chars[1], chars[2], chars[3], chars[4])
end
