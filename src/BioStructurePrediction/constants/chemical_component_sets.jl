"""
Sets of chemical component codes for classification.
"""

# Standard CCD codes for water
const WATER_COMPONENTS = Set{String}(["HOH", "WAT", "H2O", "DOD", "D2O"])

# Standard CCD codes for common ions
const ION_COMPONENTS = Set{String}([
    "ZN", "CA", "MG", "MN", "FE", "CO", "NI", "CU", "NA", "K",
    "CL", "BR", "I", "F", "PO4", "SO4", "NO3", "ACT", "EDO",
    "GOL", "PEG", "MPD", "BME",
])

# CCD codes that represent protein post-translational modifications (common)
const PTM_COMPONENTS = Set{String}([
    "SEP", "TPO", "PTR", "ALY", "MLY", "MLZ", "LLP", "KCX",
    "TYS", "HYP", "HY3", "OCS", "CSO", "CME", "MSE", "SEC",
    "FME", "PHL", "SAC", "MHO",
])

# CCD codes that represent RNA modifications (common)
const RNA_MOD_COMPONENTS = Set{String}([
    "2MG", "H2U", "5MC", "M2G", "7MG", "5MU", "PSU",
    "OMG", "OMC", "OMA", "OMU", "1MA", "T6A", "YYG",
])

# CCD codes that represent DNA modifications (common)
const DNA_MOD_COMPONENTS = Set{String}([
    "6OG", "5HC", "8OG", "5FC",
])

# CCD codes for common nucleotide cofactors
const NUCLEOTIDE_COFACTORS = Set{String}([
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "CTP", "CDP", "CMP",
    "UTP", "UDP", "UMP", "NAD", "NADH", "NADP", "FAD", "FMN",
    "COA", "SAH", "SAM",
])

# CCD codes for common drug-like ligands frequently in PDB
const COMMON_LIGANDS = Set{String}([
    "HEM", "ANP", "PO4", "SO4", "GOL", "MPD", "EDO",
    "FMT", "ACT", "ACE", "MES", "PEG", "IPA", "TRS",
])

"""
    classify_component(ccd_code::String) -> Symbol

Classify a CCD code as :water, :ion, :ptm, :rna_mod, :dna_mod, :nucleotide, :ligand, or :unknown.
"""
function classify_component(ccd_code::String)::Symbol
    ccd_code in WATER_COMPONENTS       && return :water
    ccd_code in ION_COMPONENTS         && return :ion
    ccd_code in PTM_COMPONENTS         && return :ptm
    ccd_code in RNA_MOD_COMPONENTS     && return :rna_mod
    ccd_code in DNA_MOD_COMPONENTS     && return :dna_mod
    ccd_code in NUCLEOTIDE_COFACTORS   && return :nucleotide
    ccd_code in COMMON_LIGANDS         && return :ligand
    return :unknown
end

"""
    is_water(ccd_code::String) -> Bool
"""
is_water(ccd_code::String)::Bool = ccd_code in WATER_COMPONENTS

"""
    is_ion(ccd_code::String) -> Bool
"""
is_ion(ccd_code::String)::Bool = ccd_code in ION_COMPONENTS
