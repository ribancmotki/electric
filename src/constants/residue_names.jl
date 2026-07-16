"""
residue_names.jl — residue name constants and conversion utilities.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Protein types
# ──────────────────────────────────────────────────────────────────────────────

const PROTEIN_TYPES = (
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
)

const UNK  = "UNK"
const GAP  = "-"
const UNL  = "UNL"
const MSE  = "MSE"

const PROTEIN_TYPES_WITH_UNKNOWN = (PROTEIN_TYPES..., UNK)

const PROTEIN_TYPES_ONE_LETTER = (
    "A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V",
)

const PROTEIN_COMMON_THREE_TO_ONE = Dict{String,String}(
    "ALA"=>"A","ARG"=>"R","ASN"=>"N","ASP"=>"D","CYS"=>"C",
    "GLN"=>"Q","GLU"=>"E","GLY"=>"G","HIS"=>"H","ILE"=>"I",
    "LEU"=>"L","LYS"=>"K","MET"=>"M","PHE"=>"F","PRO"=>"P",
    "SER"=>"S","THR"=>"T","TRP"=>"W","TYR"=>"Y","VAL"=>"V",
)

const PROTEIN_COMMON_ONE_TO_THREE = Dict{String,String}(
    v => k for (k, v) in PROTEIN_COMMON_THREE_TO_ONE
)

# ──────────────────────────────────────────────────────────────────────────────
# Nucleic acid types
# ──────────────────────────────────────────────────────────────────────────────

const RNA_TYPES  = ("A",  "G",  "C",  "U")
const DNA_TYPES  = ("DA", "DG", "DC", "DT")
const UNK_RNA    = "N"
const UNK_DNA    = "DN"
const UNK_NUCLEIC_ONE_LETTER = "N"

const NUCLEIC_TYPES = (RNA_TYPES..., DNA_TYPES...)
const NUCLEIC_TYPES_WITH_2_UNKS = (NUCLEIC_TYPES..., UNK_RNA, UNK_DNA)
const NUCLEIC_TYPES_WITH_UNKNOWN = NUCLEIC_TYPES_WITH_2_UNKS

const DNA_COMMON_ONE_TO_TWO = Dict{String,String}(
    "A"=>"DA","G"=>"DG","C"=>"DC","T"=>"DT",
)

const RNA_COMMON_ONE_TO_THREE = Dict{String,String}(
    "A"=>"A","G"=>"G","C"=>"C","U"=>"U",
)

# ──────────────────────────────────────────────────────────────────────────────
# Water, unknown, special types
# ──────────────────────────────────────────────────────────────────────────────

const WATER_TYPES   = ("HOH","DOD")
const UNKNOWN_TYPES = ("UNK","N","DN","UNL")

# ──────────────────────────────────────────────────────────────────────────────
# Combined polymer type ordering (for featurisation)
# ──────────────────────────────────────────────────────────────────────────────

const POLYMER_TYPES_WITH_UNKNOWN_AND_GAP = (
    PROTEIN_TYPES_WITH_UNKNOWN...,
    GAP,
    NUCLEIC_TYPES_WITH_2_UNKS...,
)

const POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP = Dict{String,Int}(
    t => i for (i, t) in enumerate(POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)
)

const POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP = length(POLYMER_TYPES_WITH_UNKNOWN_AND_GAP)  # 31

# ──────────────────────────────────────────────────────────────────────────────
# CCD 3-letter to 1-letter mapping (comprehensive)
# ──────────────────────────────────────────────────────────────────────────────

const CCD_NAME_TO_ONE_LETTER = Dict{String,String}(
    # Standard amino acids
    "ALA"=>"A","ARG"=>"R","ASN"=>"N","ASP"=>"D","CYS"=>"C",
    "GLN"=>"Q","GLU"=>"E","GLY"=>"G","HIS"=>"H","ILE"=>"I",
    "LEU"=>"L","LYS"=>"K","MET"=>"M","PHE"=>"F","PRO"=>"P",
    "SER"=>"S","THR"=>"T","TRP"=>"W","TYR"=>"Y","VAL"=>"V",
    "UNK"=>"X",
    # MSE = selenomethionine -> M
    "MSE"=>"M",
    # Modified amino acids
    "SEP"=>"S","TPO"=>"T","PTR"=>"Y","CSO"=>"C","LLP"=>"K",
    "KCX"=>"K","CSD"=>"C","CME"=>"C","OCS"=>"C","MLY"=>"K",
    "HYP"=>"P","5HP"=>"E","TYS"=>"Y","TYI"=>"Y","ALY"=>"K",
    "NEP"=>"H","HIC"=>"H","SMC"=>"C","SCH"=>"C","CSX"=>"C",
    "CSS"=>"C","CCS"=>"C","2LU"=>"L","MVA"=>"V","BMT"=>"T",
    "NH2"=>"N","MK8"=>"L","M3L"=>"K","MLZ"=>"K","M2L"=>"K",
    "QPA"=>"C","SEB"=>"S","SNN"=>"N","DHA"=>"S","CGA"=>"E",
    "AGM"=>"R","LYR"=>"K","MYK"=>"K","ACL"=>"R","PR3"=>"C",
    "GPR"=>"G","SAR"=>"G","DAL"=>"A","DSN"=>"S","DTH"=>"T",
    "DCY"=>"C","DPR"=>"P","DHI"=>"H","DAR"=>"R","DLY"=>"K",
    "DTR"=>"W","DSG"=>"N","DAS"=>"D","DGL"=>"E","DGN"=>"Q",
    "DIL"=>"I","DVA"=>"V","DLE"=>"L","DPN"=>"F","DTY"=>"Y",
    "MED"=>"M","FME"=>"M","CXM"=>"M","OMT"=>"M","SME"=>"M",
    "MHO"=>"M","CAF"=>"C","CAY"=>"C","CY0"=>"C","CY1"=>"C",
    "CY3"=>"C","CY4"=>"C","CYD"=>"C","CYF"=>"C","CYG"=>"C",
    "CYM"=>"C","CYQ"=>"C","CYW"=>"C","PCA"=>"E","GGL"=>"E",
    "GL3"=>"G","CGU"=>"E","4HY"=>"P","DHE"=>"H","HIS1"=>"H",
    "3AH"=>"H","HSD"=>"H","HSE"=>"H","HSP"=>"H","MHS"=>"H",
    "CLP"=>"C","CLH"=>"K","EFC"=>"C","QCS"=>"C","FGA"=>"E",
    "SLZ"=>"K","SHR"=>"K","YCM"=>"C","3MY"=>"Y","NIY"=>"Y",
    "PAQ"=>"Y","TYQ"=>"Y","TYB"=>"Y","IYR"=>"Y","TYX"=>"Y",
    "STY"=>"Y","OMY"=>"Y","WLU"=>"L","NLN"=>"L","NLO"=>"L",
    "MLE"=>"L","NLE"=>"L","MLL"=>"L","LYZ"=>"K","LYX"=>"K",
    "LYN"=>"K","LYM"=>"K","LYF"=>"K","KPI"=>"K","HZP"=>"P",
    "P1L"=>"C","TYY"=>"Y","11W"=>"X",
    # Nucleotides (RNA)
    "A"=>"A","G"=>"G","C"=>"C","U"=>"U","N"=>"N",
    # Nucleotides (DNA)
    "DA"=>"A","DG"=>"G","DC"=>"C","DT"=>"T","DN"=>"N",
    "DU"=>"U",
    # Modified nucleotides RNA
    "PSU"=>"U","OMU"=>"U","OMG"=>"G","OMA"=>"A","OMC"=>"C",
    "1MA"=>"A","2MA"=>"A","M2G"=>"G","7MG"=>"G","5MC"=>"C",
    "5MU"=>"U","H2U"=>"U","4SU"=>"U","QUO"=>"G","YYG"=>"G",
    "I"=>"G","A2M"=>"A","CCC"=>"C","G7M"=>"G","MA6"=>"A",
    "6MZ"=>"A","2MG"=>"G","M2G"=>"G","OMG"=>"G","7MG"=>"G",
    "1MG"=>"G","1MC"=>"C","3MC"=>"C","5OC"=>"C","IC"=>"C",
    "BRU"=>"U","5BU"=>"U","IU"=>"U","CMR"=>"C","UMS"=>"U",
    "SSU"=>"U","T6A"=>"A","MIA"=>"A","FHU"=>"U","FMU"=>"U",
    # Water / gap / unknown
    "HOH"=>"X","DOD"=>"X",
    "UNL"=>"X",
    "-"=>"-",
)

# ──────────────────────────────────────────────────────────────────────────────
# Lookup functions
# ──────────────────────────────────────────────────────────────────────────────

"""
    letters_three_to_one(restype::String; default::String="X") -> String

Map a CCD three-letter code to its single-letter code.
Uses CCD_NAME_TO_ONE_LETTER; returns `default` if not found.
"""
function letters_three_to_one(restype::String; default::String="X")::String
    return get(CCD_NAME_TO_ONE_LETTER, restype, default)
end

"""
    protein_sequence_to_residues(seq::String) -> Vector{String}

Convert a single-letter protein sequence to a vector of three-letter CCD codes.
Unknown letters map to "UNK".
"""
function protein_sequence_to_residues(seq::String)::Vector{String}
    result = Vector{String}(undef, length(seq))
    for (i, ch) in enumerate(seq)
        s = string(ch)
        result[i] = get(PROTEIN_COMMON_ONE_TO_THREE, s, UNK)
    end
    return result
end

"""
    rna_sequence_to_residues(seq::String) -> Vector{String}

Convert a single-letter RNA sequence to a vector of CCD codes.
Unknown letters map to "N" (unknown RNA).
"""
function rna_sequence_to_residues(seq::String)::Vector{String}
    valid = Set(["A","G","C","U"])
    result = Vector{String}(undef, length(seq))
    for (i, ch) in enumerate(seq)
        s = string(uppercase(ch))
        result[i] = s in valid ? s : UNK_RNA
    end
    return result
end

"""
    dna_sequence_to_residues(seq::String) -> Vector{String}

Convert a single-letter DNA sequence to a vector of CCD codes.
Unknown letters map to "DN" (unknown DNA).
"""
function dna_sequence_to_residues(seq::String)::Vector{String}
    result = Vector{String}(undef, length(seq))
    for (i, ch) in enumerate(seq)
        s = string(uppercase(ch))
        result[i] = get(DNA_COMMON_ONE_TO_TWO, s, UNK_DNA)
    end
    return result
end

"""
    residue_type_index(restype::String) -> Int

Return 1-based index of restype in POLYMER_TYPES_WITH_UNKNOWN_AND_GAP.
Returns the index of UNK if not found.
"""
function residue_type_index(restype::String)::Int
    return get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, restype,
               get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, UNK, 1))
end
