"""
structure_tables.jl — Atom-level table schema for Structure data.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Column specs for the atom-level table
# ──────────────────────────────────────────────────────────────────────────────

const ATOM_TABLE_COLUMNS = Dict{String,Type}(
    "atom_name"    => String,
    "atom_element" => String,
    "res_name"     => String,
    "res_id"       => Int,
    "chain_id"     => String,
    "chain_type"   => String,
    "atom_x"       => Float32,
    "atom_y"       => Float32,
    "atom_z"       => Float32,
    "atom_b_factor"    => Float32,
    "atom_occupancy"   => Float32,
)

"""
    make_empty_atom_table() -> Table

Create an empty atom table with the standard column schema.
"""
function make_empty_atom_table()::Table
    return empty_table(ATOM_TABLE_COLUMNS)
end

"""
    make_atom_table(;
        atom_name, atom_element, res_name, res_id, chain_id, chain_type,
        atom_x, atom_y, atom_z, atom_b_factor, atom_occupancy
    ) -> Table

Construct an atom table from parallel arrays.
"""
function make_atom_table(;
    atom_name::Vector{String},
    atom_element::Vector{String},
    res_name::Vector{String},
    res_id::Vector{Int},
    chain_id::Vector{String},
    chain_type::Vector{String},
    atom_x::Vector{Float32},
    atom_y::Vector{Float32},
    atom_z::Vector{Float32},
    atom_b_factor::Union{Vector{Float32},Nothing}   = nothing,
    atom_occupancy::Union{Vector{Float32},Nothing}  = nothing,
)::Table
    n = length(atom_name)
    cols = Dict{String,AbstractVector}(
        "atom_name"    => atom_name,
        "atom_element" => atom_element,
        "res_name"     => res_name,
        "res_id"       => res_id,
        "chain_id"     => chain_id,
        "chain_type"   => chain_type,
        "atom_x"       => atom_x,
        "atom_y"       => atom_y,
        "atom_z"       => atom_z,
        "atom_b_factor"   => atom_b_factor === nothing ? fill(0f0, n) : atom_b_factor,
        "atom_occupancy"  => atom_occupancy === nothing ? fill(1f0, n) : atom_occupancy,
    )
    return Table(cols)
end

"""
    coords_from_table(t::Table) -> Matrix{Float32}

Extract (N, 3) coordinate matrix from an atom table.
"""
function coords_from_table(t::Table)::Matrix{Float32}
    n = t._len
    result = Matrix{Float32}(undef, n, 3)
    result[:, 1] = t["atom_x"]
    result[:, 2] = t["atom_y"]
    result[:, 3] = t["atom_z"]
    return result
end

"""
    table_from_arrays(;kwargs...) -> Table

Alias for make_atom_table for convenience.
"""
table_from_arrays(;kwargs...) = make_atom_table(;kwargs...)
