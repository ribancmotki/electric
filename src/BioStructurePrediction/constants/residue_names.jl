"""
Residue name mappings and classification utilities.
"""

# Mapping from non-standard amino acid codes to standard ones (used in MSA processing)
const NON_STANDARD_RESIDUE_MAP = Dict{String,String}(
    "MSE" => "MET",  # Selenomethionine
    "SEC" => "CYS",  # Selenocysteine
    "PYL" => "LYS",  # Pyrrolysine
    "HYP" => "PRO",  # Hydroxyproline
    "HY3" => "PRO",  # trans-4-hydroxy-L-proline (modification)
    "FME" => "MET",  # N-formylmethionine
    "SAC" => "SER",  # S-acetylserine
    "CME" => "CYS",  # S,S-(2-hydroxyethyl)thiocysteine
    "OCS" => "CYS",  # Cysteinesulfinic acid
    "CSO" => "CYS",  # S-hydroxycysteine
    "CSS" => "CYS",  # S-mercaptocysteine
    "MHO" => "MET",  # S-oxymethionine
    "PHL" => "PHE",  # L-phenylalanine
    "TPO" => "THR",  # Phosphothreonine
    "SEP" => "SER",  # Phosphoserine
    "PTR" => "TYR",  # O-phosphotyrosine
    "ALY" => "LYS",  # N6-acetyllysine
    "MLY" => "LYS",  # N-methylated lysine
    "MLZ" => "LYS",  # N6-methyl-lysine
    "LLP" => "LYS",  # Lysine-pyridoxal-5-phosphate
    "KCX" => "LYS",  # N6-carboxylysine
    "TYS" => "TYR",  # Sulfotyrosine
    "2MG" => "G",    # N2-methylguanosine (RNA mod)
    "H2U" => "U",    # Dihydrouridine (RNA mod)
    "5MC" => "C",    # 5-methylcytidine (RNA mod)
    "M2G" => "G",    # N2,N2-dimethylguanosine (RNA mod)
    "7MG" => "G",    # N7-methylguanosine (RNA mod)
    "5MU" => "U",    # Ribothymidine (RNA mod)
    "PSU" => "U",    # Pseudouridine (RNA mod)
    "OMG" => "G",    # O2'-methylguanosine (RNA mod)
    "OMC" => "C",    # O2'-methylcytidine (RNA mod)
    "OMA" => "A",    # O2'-methyladenosine (RNA mod)
    "OMU" => "U",    # O2'-methyluridine (RNA mod)
    "6OG" => "DG",   # O6-methylguanosine (DNA mod)
    "5HC" => "DC",   # 5-formylcytosine (DNA mod)
)

# RNA base one-letter codes
const RNA_ONE_LETTER = Dict{Char,String}(
    'A' => "A", 'U' => "U", 'G' => "G", 'C' => "C",
)

# DNA base one-letter codes
const DNA_ONE_LETTER = Dict{Char,String}(
    'A' => "DA", 'T' => "DT", 'G' => "DG", 'C' => "DC",
)

"""
    standardise_residue_name(name::String) -> String

Map a non-standard residue name to its standard equivalent.
Returns the name unchanged if it is already standard or unknown.
"""
function standardise_residue_name(name::String)::String
    return get(NON_STANDARD_RESIDUE_MAP, name, name)
end

"""
    is_standard_amino_acid(name::String) -> Bool

Return true if name is one of the 20 standard amino acids.
"""
function is_standard_amino_acid(name::String)::Bool
    return name in AMINO_ACID_TYPES
end

"""
    is_rna_residue(name::String) -> Bool

Return true if name is a standard RNA nucleotide.
"""
function is_rna_residue(name::String)::Bool
    return name in RNA_TYPES
end

"""
    is_dna_residue(name::String) -> Bool

Return true if name is a standard DNA nucleotide.
"""
function is_dna_residue(name::String)::Bool
    return name in DNA_TYPES
end

"""
    protein_sequence_to_residues(seq::String) -> Vector{String}

Convert a one-letter protein sequence string to a vector of three-letter residue codes.
Unknown amino acids are mapped to 'UNK'.
"""
function protein_sequence_to_residues(seq::String)::Vector{String}
    return [get(ONE_LETTER_TO_THREE, c, "UNK") for c in uppercase(seq)]
end

"""
    rna_sequence_to_residues(seq::String) -> Vector{String}

Convert a one-letter RNA sequence string to a vector of residue codes.
"""
function rna_sequence_to_residues(seq::String)::Vector{String}
    return [get(RNA_ONE_LETTER, c, "UNK") for c in uppercase(seq)]
end

"""
    dna_sequence_to_residues(seq::String) -> Vector{String}

Convert a one-letter DNA sequence string to a vector of residue codes.
"""
function dna_sequence_to_residues(seq::String)::Vector{String}
    return [get(DNA_ONE_LETTER, c, "UNK") for c in uppercase(seq)]
end

"""
    residue_type_index(residue_name::String) -> Int

Return the canonical index (1-based) of a residue in POLYMER_RESIDUE_TYPES,
or length(POLYMER_RESIDUE_TYPES)+1 for unknown residues.
"""
function residue_type_index(residue_name::String)::Int
    idx = findfirst(==(residue_name), POLYMER_RESIDUE_TYPES)
    return idx === nothing ? length(POLYMER_RESIDUE_TYPES) + 1 : idx
end
