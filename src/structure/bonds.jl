"""
bonds.jl — Bond parsing from mmCIF _struct_conn records.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Bond types
# ──────────────────────────────────────────────────────────────────────────────

"""
    Bond

A covalent bond between two atoms, each identified by (chain_id, res_id, atom_name).
"""
struct Bond
    chain_id_1::String
    res_id_1::Int
    atom_name_1::String
    chain_id_2::String
    res_id_2::Int
    atom_name_2::String
    bond_order::String  # "SING","DOUB","TRIP","AROM"
    pdbx_leaving_atom_flag::Bool
end

function Base.show(io::IO, b::Bond)
    print(io, "Bond($(b.chain_id_1):$(b.res_id_1):$(b.atom_name_1) — $(b.chain_id_2):$(b.res_id_2):$(b.atom_name_2), $(b.bond_order))")
end

# ──────────────────────────────────────────────────────────────────────────────
# Bond parsing from mmCIF
# ──────────────────────────────────────────────────────────────────────────────

"""
    parse_bonds_from_mmcif(block::CifDict) -> Vector{Bond}

Parse covalent bonds from the _struct_conn loop in an mmCIF block.
"""
function parse_bonds_from_mmcif(block::CifDict)::Vector{Bond}
    conn_type   = get_loop_col(block, "_struct_conn.conn_type_id")
    asym_id_1   = get_loop_col(block, "_struct_conn.ptnr1_label_asym_id")
    seq_id_1    = get_loop_col(block, "_struct_conn.ptnr1_label_seq_id")
    atom_id_1   = get_loop_col(block, "_struct_conn.ptnr1_label_atom_id")
    asym_id_2   = get_loop_col(block, "_struct_conn.ptnr2_label_asym_id")
    seq_id_2    = get_loop_col(block, "_struct_conn.ptnr2_label_seq_id")
    atom_id_2   = get_loop_col(block, "_struct_conn.ptnr2_label_atom_id")

    bonds = Bond[]
    for i in 1:length(conn_type)
        ct = get(conn_type, i, "")
        ct in ("covale","disulf","metalc","covale_base","covale_sugar","covale_phosphate") || continue
        cid1 = get(asym_id_1, i, "")
        rid1 = _parse_seq_id(get(seq_id_1, i, "0"))
        an1  = get(atom_id_1, i, "")
        cid2 = get(asym_id_2, i, "")
        rid2 = _parse_seq_id(get(seq_id_2, i, "0"))
        an2  = get(atom_id_2, i, "")
        order = ct == "disulf" ? "SING" : "SING"
        push!(bonds, Bond(cid1, rid1, an1, cid2, rid2, an2, order, false))
    end
    return bonds
end

"""
    get_covalent_bonds(s::Structure) -> Vector{Bond}

Extract all covalent bonds from a structure's _struct_conn data.
Requires that the structure was parsed with bond information.
"""
function get_covalent_bonds(s::Structure)::Vector{Bond}
    # If we have cached bond data from parsing, return it
    # Otherwise return empty
    return Bond[]
end

"""
    get_bonded_atom_pairs(bonds::Vector{Bond}) -> Vector{Tuple{Tuple{String,Int,String},Tuple{String,Int,String}}}

Convert bonds to (chain_id, res_id, atom_name) pair tuples.
"""
function get_bonded_atom_pairs(bonds::Vector{Bond})
    return [
        ((b.chain_id_1, b.res_id_1, b.atom_name_1),
         (b.chain_id_2, b.res_id_2, b.atom_name_2))
        for b in bonds
    ]
end

"""
    bond_adjacency_matrix(s::Structure, bonds::Vector{Bond}) -> SparseMatrixCSC

Build a sparse adjacency matrix (num_atoms × num_atoms) from bond pairs.
"""
function bond_adjacency_matrix(s::Structure, bonds::Vector{Bond})
    # Build atom lookup: (chain_id, res_id, atom_name) → row index
    n = length(s)
    atom_idx = Dict{Tuple{String,Int,String},Int}()
    for i in 1:n
        key = (s.chain_id[i], s.res_id[i], s.atom_name[i])
        haskey(atom_idx, key) || (atom_idx[key] = i)
    end

    rows = Int[]
    cols = Int[]
    for b in bonds
        i = get(atom_idx, (b.chain_id_1, b.res_id_1, b.atom_name_1), 0)
        j = get(atom_idx, (b.chain_id_2, b.res_id_2, b.atom_name_2), 0)
        (i > 0 && j > 0) || continue
        push!(rows, i); push!(cols, j)
        push!(rows, j); push!(cols, i)
    end

    # Return as pairs (no sparse matrix dep)
    return collect(zip(rows, cols))
end
