"""
Standard atom type definitions for proteins, nucleic acids, and small molecules.
"""

# Standard amino acid atom order (up to 37 heavy atoms per residue)
# Positions 0-36 in the dense atom representation
const ATOM_ORDER = Dict{String,Vector{String}}(
    "ALA" => ["N","CA","C","O","CB","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "ARG" => ["N","CA","C","O","CB","CG","CD","NE","CZ","NH1","NH2","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "ASN" => ["N","CA","C","O","CB","CG","OD1","ND2","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "ASP" => ["N","CA","C","O","CB","CG","OD1","OD2","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "CYS" => ["N","CA","C","O","CB","SG","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "GLN" => ["N","CA","C","O","CB","CG","CD","OE1","NE2","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "GLU" => ["N","CA","C","O","CB","CG","CD","OE1","OE2","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "GLY" => ["N","CA","C","O","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "HIS" => ["N","CA","C","O","CB","CG","ND1","CD2","CE1","NE2","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "ILE" => ["N","CA","C","O","CB","CG1","CG2","CD1","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "LEU" => ["N","CA","C","O","CB","CG","CD1","CD2","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "LYS" => ["N","CA","C","O","CB","CG","CD","CE","NZ","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "MET" => ["N","CA","C","O","CB","CG","SD","CE","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "PHE" => ["N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "PRO" => ["N","CA","C","O","CB","CG","CD","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "SER" => ["N","CA","C","O","CB","OG","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "THR" => ["N","CA","C","O","CB","OG1","CG2","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "TRP" => ["N","CA","C","O","CB","CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2","","","","","","","","","","","","","","","","","","","","","","",""],
    "TYR" => ["N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ","OH","","","","","","","","","","","","","","","","","","","","","","","","",""],
    "VAL" => ["N","CA","C","O","CB","CG1","CG2","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
    # RNA nucleotides
    "A"   => ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N9","C8","N7","C5","C6","N6","N1","C2","N3","C4","","","","","","","","","","","","","","",""],
    "U"   => ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N1","C2","O2","N3","C4","O4","C5","C6","","","","","","","","","","","","","","","","",""],
    "G"   => ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N9","C8","N7","C5","C6","O6","N1","C2","N2","N3","C4","","","","","","","","","","","","","",""],
    "C"   => ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N1","C2","O2","N3","C4","N4","C5","C6","","","","","","","","","","","","","","","","",""],
    # DNA nucleotides
    "DA"  => ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'","N9","C8","N7","C5","C6","N6","N1","C2","N3","C4","","","","","","","","","","","","","","","",""],
    "DT"  => ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'","N1","C2","O2","N3","C4","O4","C5","C7","C6","","","","","","","","","","","","","","","","",""],
    "DG"  => ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'","N9","C8","N7","C5","C6","O6","N1","C2","N2","N3","C4","","","","","","","","","","","","","","",""],
    "DC"  => ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","C1'","N1","C2","O2","N3","C4","N4","C5","C6","","","","","","","","","","","","","","","","","",""],
    "UNK" => ["CA","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],
)

# Mask indicating which of the 37 slots are occupied for each residue type
const STANDARD_ATOM_MASK = Dict{String,Vector{Bool}}(
    k => [!isempty(a) for a in v]
    for (k, v) in ATOM_ORDER
)

# Canonical list of 20 amino acids
const AMINO_ACID_TYPES = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                           "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]

# RNA nucleotide types
const RNA_TYPES = ["A","U","G","C"]

# DNA nucleotide types
const DNA_TYPES = ["DA","DT","DG","DC"]

# All standard polymer residue types in canonical order
const POLYMER_RESIDUE_TYPES = vcat(AMINO_ACID_TYPES, RNA_TYPES, DNA_TYPES)

# One-letter amino acid code mapping
const ONE_LETTER_TO_THREE = Dict{Char,String}(
    'A' => "ALA", 'R' => "ARG", 'N' => "ASN", 'D' => "ASP", 'C' => "CYS",
    'Q' => "GLN", 'E' => "GLU", 'G' => "GLY", 'H' => "HIS", 'I' => "ILE",
    'L' => "LEU", 'K' => "LYS", 'M' => "MET", 'F' => "PHE", 'P' => "PRO",
    'S' => "SER", 'T' => "THR", 'W' => "TRP", 'Y' => "TYR", 'V' => "VAL",
    'X' => "UNK",
)

const THREE_LETTER_TO_ONE = Dict{String,Char}(v => k for (k, v) in ONE_LETTER_TO_THREE)

# Element index for standard biological elements (1-indexed, 0 = unknown)
const ELEMENT_ORDER = Dict{String,Int}(
    "H"  =>  1, "C"  =>  2, "N"  =>  3, "O"  =>  4, "F"  =>  5,
    "P"  =>  6, "S"  =>  7, "Cl" =>  8, "Se" =>  9, "Br" => 10,
    "I"  => 11, "Fe" => 12, "Co" => 13, "Cu" => 14, "Zn" => 15,
    "Mn" => 16, "Mg" => 17, "Ca" => 18, "Na" => 19, "K"  => 20,
    "B"  => 21, "Si" => 22, "As" => 23,
)
const NUM_ELEMENTS = 128  # Covers full periodic table

# Total number of atom slots per token in the dense representation
const NUM_ATOM_SLOTS = 37

# Van der Waals radii in Angstroms for clash detection
const VDW_RADII = Dict{String,Float32}(
    "H"  => 1.20f0, "C"  => 1.70f0, "N"  => 1.55f0, "O"  => 1.52f0,
    "S"  => 1.80f0, "P"  => 1.80f0, "F"  => 1.47f0, "Cl" => 1.75f0,
    "Br" => 1.85f0, "I"  => 1.98f0, "Se" => 1.90f0, "Fe" => 1.80f0,
    "Co" => 1.80f0, "Cu" => 1.40f0, "Zn" => 1.39f0, "Mg" => 1.73f0,
    "Mn" => 1.73f0, "Ca" => 2.31f0, "Na" => 2.27f0, "K"  => 2.75f0,
)
const DEFAULT_VDW_RADIUS = 1.70f0

"""
    get_vdw_radius(element::String) -> Float32

Return the Van der Waals radius for the given element symbol.
"""
function get_vdw_radius(element::String)::Float32
    return get(VDW_RADII, element, DEFAULT_VDW_RADIUS)
end

"""
    get_element_index(element::String) -> Int

Return the 0-based element index. Returns 0 for unknown elements.
"""
function get_element_index(element::String)::Int
    idx = get(ELEMENT_ORDER, element, nothing)
    return idx === nothing ? 0 : idx
end

"""
    atom_name_to_chars(name::String) -> Vector{Int32}

Encode an atom name (up to 4 chars) as a vector of 4 integer character codes.
Pads with zeros for shorter names.
"""
function atom_name_to_chars(name::String)::Vector{Int32}
    chars = zeros(Int32, 4)
    for (i, c) in enumerate(name)
        if i > 4
            break
        end
        chars[i] = Int32(c)
    end
    return chars
end

"""
    get_atom_index(residue_type::String, atom_name::String) -> Union{Int,Nothing}

Return the 1-based index of atom_name in the standard atom order for residue_type,
or nothing if not found.
"""
function get_atom_index(residue_type::String, atom_name::String)::Union{Int,Nothing}
    atoms = get(ATOM_ORDER, residue_type, nothing)
    if atoms === nothing
        return nothing
    end
    idx = findfirst(==(atom_name), atoms)
    return idx
end
