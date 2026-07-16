"""
Column-oriented table implementation for structure data.
"""

"""
    StructureTable

A column-oriented table with named, typed columns. Each column is a Vector.
"""
mutable struct StructureTable
    columns::Dict{Symbol,Vector}
    column_order::Vector{Symbol}
    n_rows::Int

    function StructureTable()
        new(Dict{Symbol,Vector}(), Symbol[], 0)
    end

    function StructureTable(columns::Dict{Symbol,Vector}, column_order::Vector{Symbol})
        n = isempty(column_order) ? 0 : length(columns[first(column_order)])
        new(columns, column_order, n)
    end
end

"""
    add_column!(t::StructureTable, name::Symbol, data::Vector)

Add or replace a column in the table. The data length must match existing rows
(or the table must be empty).
"""
function add_column!(t::StructureTable, name::Symbol, data::Vector)
    if t.n_rows == 0 && isempty(t.column_order)
        t.n_rows = length(data)
    elseif length(data) != t.n_rows
        error("Column '$name' has $(length(data)) rows, expected $(t.n_rows)")
    end
    if !haskey(t.columns, name)
        push!(t.column_order, name)
    end
    t.columns[name] = data
end

"""
    get_column(t::StructureTable, name::Symbol) -> Vector

Retrieve a column by name. Raises KeyError if not found.
"""
function get_column(t::StructureTable, name::Symbol)::Vector
    haskey(t.columns, name) || error("Column '$name' not found in StructureTable")
    return t.columns[name]
end

"""
    has_column(t::StructureTable, name::Symbol) -> Bool

Return true if the column exists.
"""
function has_column(t::StructureTable, name::Symbol)::Bool
    return haskey(t.columns, name)
end

"""
    nrows(t::StructureTable) -> Int

Return the number of rows in the table.
"""
function nrows(t::StructureTable)::Int
    return t.n_rows
end

"""
    ncols(t::StructureTable) -> Int

Return the number of columns.
"""
function ncols(t::StructureTable)::Int
    return length(t.column_order)
end

"""
    column_names(t::StructureTable) -> Vector{Symbol}

Return the column names in insertion order.
"""
function column_names(t::StructureTable)::Vector{Symbol}
    return copy(t.column_order)
end

"""
    filter_rows(t::StructureTable, mask::AbstractVector{Bool}) -> StructureTable

Return a new StructureTable containing only the rows where mask is true.
"""
function filter_rows(t::StructureTable, mask::AbstractVector{Bool})::StructureTable
    length(mask) == t.n_rows || error("Mask length $(length(mask)) != table rows $(t.n_rows)")
    new_cols = Dict{Symbol,Vector}()
    for name in t.column_order
        col = t.columns[name]
        new_cols[name] = col[mask]
    end
    return StructureTable(new_cols, copy(t.column_order))
end

"""
    concat(tables::Vector{StructureTable}) -> StructureTable

Concatenate a list of StructureTables row-wise.
All tables must have the same column names.
"""
function concat(tables::Vector{StructureTable})::StructureTable
    isempty(tables) && return StructureTable()
    ref_cols = tables[1].column_order
    for t in tables[2:end]
        if Set(t.column_order) != Set(ref_cols)
            error("Cannot concat tables with different column sets")
        end
    end
    new_cols = Dict{Symbol,Vector}()
    for name in ref_cols
        new_cols[name] = vcat([t.columns[name] for t in tables]...)
    end
    return StructureTable(new_cols, copy(ref_cols))
end

"""
    get_row(t::StructureTable, i::Int) -> Dict{Symbol,Any}

Return the i-th row as a Dict.
"""
function get_row(t::StructureTable, i::Int)::Dict{Symbol,Any}
    row = Dict{Symbol,Any}()
    for name in t.column_order
        row[name] = t.columns[name][i]
    end
    return row
end

"""
    sort_table(t::StructureTable, by::Symbol) -> StructureTable

Return a new StructureTable sorted by the given column.
"""
function sort_table(t::StructureTable, by::Symbol)::StructureTable
    col = get_column(t, by)
    perm = sortperm(col)
    new_cols = Dict{Symbol,Vector}()
    for name in t.column_order
        new_cols[name] = t.columns[name][perm]
    end
    return StructureTable(new_cols, copy(t.column_order))
end
