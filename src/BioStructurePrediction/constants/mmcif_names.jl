"""
mmCIF data category and field name constants.
"""

# _atom_site field names in canonical order for output
const ATOM_SITE_FIELDS = [
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "auth_seq_id",
    "auth_asym_id",
    "pdbx_PDB_model_num",
]

# _entity fields
const ENTITY_FIELDS = ["id", "type", "pdbx_description", "formula_weight"]

# _entity_poly fields
const ENTITY_POLY_FIELDS = [
    "entity_id", "type", "pdbx_seq_one_letter_code",
    "pdbx_seq_one_letter_code_can", "pdbx_strand_id"
]

# _struct_asym fields
const STRUCT_ASYM_FIELDS = ["id", "entity_id", "pdbx_blank_PDB_chainid_flag", "details"]

# _chem_comp fields
const CHEM_COMP_FIELDS = ["id", "type", "mon_nstd_flag", "name", "formula", "formula_weight"]

# _pdbx_audit_revision_history fields
const AUDIT_REVISION_HISTORY_FIELDS = ["ordinal", "data_content_type", "major_revision", "minor_revision", "revision_date"]

# _pdbx_struct_assembly fields
const STRUCT_ASSEMBLY_FIELDS = ["id", "details", "oligomeric_details", "oligomeric_count"]

# _pdbx_struct_assembly_gen fields
const STRUCT_ASSEMBLY_GEN_FIELDS = ["assembly_id", "oper_expression", "asym_id_list"]

# _pdbx_struct_oper_list fields
const STRUCT_OPER_LIST_FIELDS = [
    "id", "type", "name",
    "matrix[1][1]", "matrix[1][2]", "matrix[1][3]",
    "matrix[2][1]", "matrix[2][2]", "matrix[2][3]",
    "matrix[3][1]", "matrix[3][2]", "matrix[3][3]",
    "vector[1]", "vector[2]", "vector[3]",
]

# _chem_comp_atom fields
const CHEM_COMP_ATOM_FIELDS = [
    "comp_id", "atom_id", "alt_atom_id", "type_symbol",
    "charge", "pdbx_model_Cartn_x_ideal", "pdbx_model_Cartn_y_ideal", "pdbx_model_Cartn_z_ideal",
    "model_Cartn_x", "model_Cartn_y", "model_Cartn_z",
    "pdbx_leaving_atom_flag", "pdbx_stereo_config",
]

# _chem_comp_bond fields
const CHEM_COMP_BOND_FIELDS = ["comp_id", "atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"]

# mmCIF chain type strings
const CHAIN_TYPE_PROTEIN = "polypeptide(L)"
const CHAIN_TYPE_RNA     = "polyribonucleotide"
const CHAIN_TYPE_DNA     = "polydeoxyribonucleotide"
const CHAIN_TYPE_LIGAND  = "non-polymer"
const CHAIN_TYPE_ION     = "non-polymer"

# mmCIF entity types
const ENTITY_TYPE_POLYMER   = "polymer"
const ENTITY_TYPE_NON_POLYMER = "non-polymer"
const ENTITY_TYPE_WATER     = "water"

# Bond order strings
const BOND_ORDER_SINGLE   = "SING"
const BOND_ORDER_DOUBLE   = "DOUB"
const BOND_ORDER_TRIPLE   = "TRIP"
const BOND_ORDER_AROMATIC = "AROM"

const BOND_ORDER_MAP = Dict{String,Int}(
    BOND_ORDER_SINGLE   => 1,
    BOND_ORDER_DOUBLE   => 2,
    BOND_ORDER_TRIPLE   => 3,
    BOND_ORDER_AROMATIC => 4,
)

# Group PDB strings
const GROUP_PDB_ATOM  = "ATOM  "
const GROUP_PDB_HETATM = "HETATM"
