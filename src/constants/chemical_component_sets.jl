"""
chemical_component_sets.jl — sets of CCD codes for glycan and ligand categories.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Glycan "other" ligands: monosaccharides and derivatives not in a polymer chain
# ──────────────────────────────────────────────────────────────────────────────

const GLYCAN_OTHER_LIGANDS = Set{String}([
    "FUC", "FUL", "GAL", "GLA", "GLC", "BGC", "GCS", "GCU", "GLA",
    "IDR", "IDS", "MAL", "MAN", "NAG", "NDG", "NGA", "SIA", "SLB",
    "XYL", "XYS", "BMA", "GNB", "FCA", "GLB", "GYP", "HBZ",
    "LAK", "LAT", "LEU", "LFR", "LMU", "LNB", "LNT", "LSX",
    "MAB", "MAG", "MAL", "MCU", "MDA", "MFB", "MFU", "MGL",
    "MQO", "MRH", "MRP", "MTT", "MXY", "MYB", "MZB",
    "NAA", "NAB", "NAL", "NAM", "NCO", "NLC", "NLX", "NNA",
    "NOX", "NPH", "NSQ", "NTO",
    "OPT", "ORP",
    "PAR", "PDX", "PGM", "PMP", "PNA", "PNJ", "POB", "PPZ",
    "PSJ", "PST",
    "QUO",
    "RAF", "RGG", "RIB", "RIP", "RLA", "RLG", "RMD", "RML",
    "RNS", "RPA",
    "SAK", "SCA", "SDY", "SGA", "SHZ", "SIA", "SIB", "SLB",
    "SMB", "SRT", "SSG", "STZ",
    "TAL", "TGA", "TGN", "TIA", "TGX", "THG",
    "UAP", "UDC",
    "VGF",
    "WAA",
    "XBP", "XLF", "XYL", "XYP", "XYS",
])

# ──────────────────────────────────────────────────────────────────────────────
# Glycan linking ligands: sugars that form glycosidic bonds (in BRANCHED entities)
# ──────────────────────────────────────────────────────────────────────────────

const GLYCAN_LINKING_LIGANDS = Set{String}([
    "FUC","FUL","GAL","GLA","GLC","BGC","MAN","NAG","NDG","NGA",
    "SIA","SLB","XYL","BMA","FCA","GNB","MAB","MAL",
    "AFL","BDP","GCP","IDS","IDR",
])

# ──────────────────────────────────────────────────────────────────────────────
# Common drug-like ligands frequently found in PDB structures
# ──────────────────────────────────────────────────────────────────────────────

const COMMON_LIGANDS = Set{String}([
    "ATP","ADP","AMP","ADP","GTP","GDP","GMP",
    "CTP","CDP","CMP","UTP","UDP","UMP",
    "NAD","NDP","NAP","FAD","FMN",
    "HEM","HEA","HEB","HEC",
    "ZN","MG","CA","FE","MN","CU","CO","NI","K","NA",
    "EDO","GOL","PEG","PG4","PG5","MPD","MES","TRS","HEPES","PO4","SO4","CL",
    "ATP","ADP","SAM","SAH","COA","ACO","MLA","FUM","SUC",
])

# ──────────────────────────────────────────────────────────────────────────────
# Water and ion sets
# ──────────────────────────────────────────────────────────────────────────────

const WATER_COMPONENT_IDS = Set{String}(["HOH","DOD","H2O"])

const ION_IDS = Set{String}([
    "ZN","MG","CA","FE","MN","CU","CO","NI","K","NA","LI","RB","CS",
    "SR","BA","RA","CD","PB","HG","TL","AU","AG","PT","PD","IR","OS",
    "RU","RH","CR","MO","W","V","VO4","PO4","SO4","CL","BR","I","F",
    "IOD","BRO","CLO","FLO","NO3","NO2","NH4","HCO3","CO3","ClO4",
])

"""
    classify_component(ccd_id::String) -> Symbol

Classify a CCD code into one of: :water, :ion, :glycan_linking,
:glycan_other, :common_ligand, :unknown_ligand.
"""
function classify_component(ccd_id::String)::Symbol
    ccd_id in WATER_COMPONENT_IDS && return :water
    ccd_id in ION_IDS && return :ion
    ccd_id in GLYCAN_LINKING_LIGANDS && return :glycan_linking
    ccd_id in GLYCAN_OTHER_LIGANDS && return :glycan_other
    ccd_id in COMMON_LIGANDS && return :common_ligand
    return :unknown_ligand
end
