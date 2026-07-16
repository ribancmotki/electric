"""
Side chain definitions and torsion angle (chi) specifications.
"""

# Chi angle definitions: residue → list of (atom1, atom2, atom3, atom4) tuples
const CHI_ANGLES = Dict{String,Vector{NTuple{4,String}}}(
    "ARG" => [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","NE"), ("CG","CD","NE","CZ")],
    "ASN" => [("N","CA","CB","CG"), ("CA","CB","CG","OD1")],
    "ASP" => [("N","CA","CB","CG"), ("CA","CB","CG","OD1")],
    "CYS" => [("N","CA","CB","SG")],
    "GLN" => [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","OE1")],
    "GLU" => [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","OE1")],
    "HIS" => [("N","CA","CB","CG"), ("CA","CB","CG","ND1")],
    "ILE" => [("N","CA","CB","CG1"), ("CA","CB","CG1","CD1")],
    "LEU" => [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "LYS" => [("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","CE"), ("CG","CD","CE","NZ")],
    "MET" => [("N","CA","CB","CG"), ("CA","CB","CG","SD"), ("CB","CG","SD","CE")],
    "PHE" => [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "PRO" => [("N","CA","CB","CG"), ("CA","CB","CG","CD")],
    "SER" => [("N","CA","CB","OG")],
    "THR" => [("N","CA","CB","OG1")],
    "TRP" => [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "TYR" => [("N","CA","CB","CG"), ("CA","CB","CG","CD1")],
    "VAL" => [("N","CA","CB","CG1")],
    # No chi angles for ALA, GLY
)

# Chiral centers per residue type (atom, neighbors in order for L-chirality check)
const CHIRAL_CENTERS = Dict{String,Vector{Tuple{String,Vector{String}}}}(
    "ALA" => [("CA", ["N","C","CB","HA"])],
    "ARG" => [("CA", ["N","C","CB","HA"]), ("CG", ["CB","CD","HG2","HG3"])],
    "ASN" => [("CA", ["N","C","CB","HA"])],
    "ASP" => [("CA", ["N","C","CB","HA"])],
    "CYS" => [("CA", ["N","C","CB","HA"])],
    "GLN" => [("CA", ["N","C","CB","HA"])],
    "GLU" => [("CA", ["N","C","CB","HA"])],
    "HIS" => [("CA", ["N","C","CB","HA"])],
    "ILE" => [("CA", ["N","C","CB","HA"]), ("CB", ["CA","CG1","CG2","HB"])],
    "LEU" => [("CA", ["N","C","CB","HA"])],
    "LYS" => [("CA", ["N","C","CB","HA"])],
    "MET" => [("CA", ["N","C","CB","HA"])],
    "PHE" => [("CA", ["N","C","CB","HA"])],
    "PRO" => [("CA", ["N","C","CB","HA"])],
    "SER" => [("CA", ["N","C","CB","HA"])],
    "THR" => [("CA", ["N","C","CB","HA"]), ("CB", ["CA","OG1","CG2","HB"])],
    "TRP" => [("CA", ["N","C","CB","HA"])],
    "TYR" => [("CA", ["N","C","CB","HA"])],
    "VAL" => [("CA", ["N","C","CB","HA"])],
)

"""
    get_chi_angles(residue_name::String) -> Vector{NTuple{4,String}}

Return the chi angle definitions for the given residue.
Returns an empty vector for residues with no chi angles (GLY, ALA, etc.).
"""
function get_chi_angles(residue_name::String)::Vector{NTuple{4,String}}
    return get(CHI_ANGLES, residue_name, NTuple{4,String}[])
end

"""
    get_chiral_centers(residue_name::String) -> Vector{Tuple{String,Vector{String}}}

Return the chiral center definitions for the given residue.
"""
function get_chiral_centers(residue_name::String)::Vector{Tuple{String,Vector{String}}}
    return get(CHIRAL_CENTERS, residue_name, Tuple{String,Vector{String}}[])
end

# Backbone atoms (always checked in chirality / frame computations)
const BACKBONE_ATOMS = ["N", "CA", "C", "O"]

# Pseudo-beta carbon positions for template embedding
# For protein: CB position (or CA for GLY)
# For RNA/DNA: C4' position
const PSEUDO_BETA_ATOM = Dict{String,String}(
    residue => (residue == "GLY" ? "CA" : "CB")
    for residue in AMINO_ACID_TYPES
)
for r in RNA_TYPES
    PSEUDO_BETA_ATOM[r] = "C4'"
end
for r in DNA_TYPES
    PSEUDO_BETA_ATOM[r] = "C4'"
end
