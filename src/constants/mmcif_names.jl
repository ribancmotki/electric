"""
mmcif_names.jl — mmCIF chain type and entity type constants.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Chain type strings as they appear in mmCIF _entity_poly.type
# ──────────────────────────────────────────────────────────────────────────────

const PROTEIN_CHAIN = "polypeptide(L)"
const RNA_CHAIN     = "polyribonucleotide"
const DNA_CHAIN     = "polydeoxyribonucleotide"
const OTHER_CHAIN   = "other"
const WATER         = "water"
const NON_POLYMER   = "non-polymer"
const BRANCHED      = "branched"

# ──────────────────────────────────────────────────────────────────────────────
# Grouped chain type constants
# ──────────────────────────────────────────────────────────────────────────────

const PEPTIDE_CHAIN_TYPES        = (PROTEIN_CHAIN,)
const NUCLEIC_ACID_CHAIN_TYPES   = (RNA_CHAIN, DNA_CHAIN, OTHER_CHAIN)
const POLYMER_CHAIN_TYPES        = (PROTEIN_CHAIN, RNA_CHAIN, DNA_CHAIN, OTHER_CHAIN)
const STANDARD_POLYMER_CHAIN_TYPES = (PROTEIN_CHAIN, RNA_CHAIN, DNA_CHAIN)
const NON_POLYMER_CHAIN_TYPES    = (NON_POLYMER, BRANCHED)
const LIGAND_CHAIN_TYPES         = (NON_POLYMER, BRANCHED)
const WATER_CHAIN_TYPES          = (WATER,)
const ALL_CHAIN_TYPES = (
    PROTEIN_CHAIN, RNA_CHAIN, DNA_CHAIN, OTHER_CHAIN,
    NON_POLYMER, BRANCHED, WATER,
)

# ──────────────────────────────────────────────────────────────────────────────
# Residue name normalization maps for non-standard polymer residues
# ──────────────────────────────────────────────────────────────────────────────

# MSE (selenomethionine) → MET
const NON_STANDARD_PROTEIN_MAP = Dict{String,String}(
    "MSE" => "MET",
    "SEP" => "SER",
    "TPO" => "THR",
    "PTR" => "TYR",
    "PCA" => "GLU",
)

const NON_STANDARD_RNA_MAP = Dict{String,String}(
    "PSU" => "U",
    "OMU" => "U",
    "5MU" => "U",
    "H2U" => "U",
    "4SU" => "U",
    "OMG" => "G",
    "7MG" => "G",
    "5MC" => "C",
    "1MA" => "A",
    "2MA" => "A",
    "M2G" => "G",
)

const NON_STANDARD_DNA_MAP = Dict{String,String}(
    "CBR" => "DC",
    "BRU" => "DU",
)

# ──────────────────────────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────────────────────────

"""
    is_standard_polymer_type(chain_type::String) -> Bool

Return true if the chain type is a standard polymer type (protein, RNA, or DNA).
"""
function is_standard_polymer_type(chain_type::String)::Bool
    return chain_type in STANDARD_POLYMER_CHAIN_TYPES
end

"""
    fix_non_standard_polymer_res(res_name::String, chain_type::String) -> String

Map non-standard residue names to their standard equivalents for the given chain type.
Returns the input unchanged if no mapping is found.
"""
function fix_non_standard_polymer_res(res_name::String, chain_type::String)::String
    if chain_type == PROTEIN_CHAIN
        return get(NON_STANDARD_PROTEIN_MAP, res_name, res_name)
    elseif chain_type == RNA_CHAIN
        return get(NON_STANDARD_RNA_MAP, res_name, res_name)
    elseif chain_type == DNA_CHAIN
        return get(NON_STANDARD_DNA_MAP, res_name, res_name)
    else
        return res_name
    end
end

"""
    is_protein_chain_type(chain_type::String) -> Bool
"""
is_protein_chain_type(chain_type::String)::Bool = chain_type == PROTEIN_CHAIN

"""
    is_rna_chain_type(chain_type::String) -> Bool
"""
is_rna_chain_type(chain_type::String)::Bool = chain_type == RNA_CHAIN

"""
    is_dna_chain_type(chain_type::String) -> Bool
"""
is_dna_chain_type(chain_type::String)::Bool = chain_type == DNA_CHAIN

"""
    is_nucleic_chain_type(chain_type::String) -> Bool
"""
is_nucleic_chain_type(chain_type::String)::Bool =
    chain_type in (RNA_CHAIN, DNA_CHAIN)

"""
    is_ligand_chain_type(chain_type::String) -> Bool
"""
is_ligand_chain_type(chain_type::String)::Bool = chain_type in LIGAND_CHAIN_TYPES

"""
    is_water_chain_type(chain_type::String) -> Bool
"""
is_water_chain_type(chain_type::String)::Bool = chain_type in WATER_CHAIN_TYPES
