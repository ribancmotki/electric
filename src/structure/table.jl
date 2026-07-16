"""
table.jl — Column-oriented Table data structure.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Table type
# ──────────────────────────────────────────────────────────────────────────────

"""
    Table

Column-oriented data structure where each column is a Vector of a fixed element type.
Supports indexing, filtering, concatenation, and iteration over rows as Dict{String,Any}.
"""
struct Table
    columns::Dict{String,AbstractVector}
    column_order::Vector{String}
    _len::Int
end

function Table(columns::Dict{String,<:AbstractVector})
    isempty(columns) && return Table(Dict{String,AbstractVector}(), String[], 0)
    lens = [length(v) for v in values(columns)]
    @assert all(==(lens[1]), lens) "All columns must have the same length, got: $lens"
    col_order = sort(collect(keys(columns)))
    return Table(Dict{String,AbstractVector}(k => v for (k,v) in columns), col_order, lens[1])
end

function Table(; kwargs...)
    cols = Dict{String,AbstractVector}()
    for (k, v) in kwargs
        cols[String(k)] = collect(v)
    end
    return Table(cols)
end

Base.length(t::Table) = t._len
Base.isempty(t::Table) = t._len == 0

function Base.show(io::IO, t::Table)
    print(io, "Table($(t._len) rows × $(length(t.columns)) cols: $(join(t.column_order, ", ")))")
end

# ──────────────────────────────────────────────────────────────────────────────
# Column access
# ──────────────────────────────────────────────────────────────────────────────

function Base.getproperty(t::Table, name::Symbol)
    if name in (:columns, :column_order, :_len)
        return getfield(t, name)
    end
    key = String(name)
    haskey(t.columns, key) && return t.columns[key]
    return getfield(t, name)
end

function Base.getindex(t::Table, key::String)
    return t.columns[key]
end

function Base.haskey(t::Table, key::String)
    return haskey(t.columns, key)
end

# ──────────────────────────────────────────────────────────────────────────────
# Row indexing
# ──────────────────────────────────────────────────────────────────────────────

"""
    table_getrows(t::Table, idxs) -> Table

Return a new Table with only the rows at indices `idxs`.
"""
function table_getrows(t::Table, idxs)::Table
    new_cols = Dict{String,AbstractVector}()
    for (k, v) in t.columns
        new_cols[k] = v[idxs]
    end
    return Table(new_cols)
end

Base.getindex(t::Table, rows::AbstractVector{<:Integer}) = table_getrows(t, rows)
Base.getindex(t::Table, mask::BitVector) = table_getrows(t, findall(mask))
Base.getindex(t::Table, mask::Vector{Bool}) = table_getrows(t, findall(mask))

# ──────────────────────────────────────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────────────────────────────────────

"""
    filter_table(t::Table; kwargs...) -> Table

Filter rows where each specified column equals the given value.
"""
function filter_table(t::Table; kwargs...)::Table
    isempty(t) && return t
    mask = trues(t._len)
    for (k, v) in kwargs
        key = String(k)
        if haskey(t.columns, key)
            col = t.columns[key]
            if v isa AbstractVector
                mask .&= in.(col, Ref(Set(v)))
            else
                mask .&= (col .== v)
            end
        end
    end
    return table_getrows(t, findall(mask))
end

# ──────────────────────────────────────────────────────────────────────────────
# Concatenation
# ──────────────────────────────────────────────────────────────────────────────

"""
    cat_tables(tables::Vector{Table}) -> Table

Concatenate multiple Tables vertically (stack rows).
All tables must have the same column names.
"""
function cat_tables(tables::Vector{Table})::Table
    isempty(tables) && return Table(Dict{String,AbstractVector}())
    tables = filter(t -> t._len > 0, tables)
    isempty(tables) && return Table(Dict{String,AbstractVector}())
    length(tables) == 1 && return tables[1]

    ref_keys = Set(tables[1].column_order)
    for t in tables[2:end]
        @assert Set(t.column_order) == ref_keys "Tables have different column sets"
    end

    new_cols = Dict{String,AbstractVector}()
    for k in tables[1].column_order
        new_cols[k] = vcat([t.columns[k] for t in tables]...)
    end
    return Table(new_cols)
end

# ──────────────────────────────────────────────────────────────────────────────
# Iteration over rows
# ──────────────────────────────────────────────────────────────────────────────

"""
    rows(t::Table) -> Vector{Dict{String,Any}}

Return all rows as a vector of dicts.
"""
function rows(t::Table)::Vector{Dict{String,Any}}
    result = Vector{Dict{String,Any}}(undef, t._len)
    for i in 1:t._len
        d = Dict{String,Any}()
        for k in t.column_order
            d[k] = t.columns[k][i]
        end
        result[i] = d
    end
    return result
end

struct RowIterator
    table::Table
end

function Base.iterate(iter::RowIterator, i::Int=1)
    i > iter.table._len && return nothing
    d = Dict{String,Any}()
    for k in iter.table.column_order
        d[k] = iter.table.columns[k][i]
    end
    return d, i + 1
end

Base.length(iter::RowIterator) = iter.table._len
Base.eltype(::RowIterator) = Dict{String,Any}

"""
    iter_rows(t::Table) -> RowIterator

Return a lazy row iterator over the table.
"""
iter_rows(t::Table) = RowIterator(t)

# ──────────────────────────────────────────────────────────────────────────────
# Sorting
# ──────────────────────────────────────────────────────────────────────────────

"""
    sort_table(t::Table, key::String; rev=false) -> Table

Sort rows by a single column.
"""
function sort_table(t::Table, key::String; rev::Bool=false)::Table
    isempty(t) && return t
    perm = sortperm(t.columns[key]; rev)
    return table_getrows(t, perm)
end

# ──────────────────────────────────────────────────────────────────────────────
# Unique values
# ──────────────────────────────────────────────────────────────────────────────

"""
    unique_values(t::Table, key::String) -> Vector

Return unique values of a column, preserving first occurrence order.
"""
function unique_values(t::Table, key::String)
    return unique(t.columns[key])
end

# ──────────────────────────────────────────────────────────────────────────────
# Construction helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    empty_table(column_specs::Dict{String,Type}) -> Table

Create an empty table with given column names and element types.
"""
function empty_table(column_specs::Dict{String,Type})::Table
    cols = Dict{String,AbstractVector}(k => Vector{T}() for (k, T) in column_specs)
    return Table(cols)
end

"""
    add_column(t::Table, name::String, values::AbstractVector) -> Table

Return a new Table with an additional column.
"""
function add_column(t::Table, name::String, values::AbstractVector)::Table
    @assert length(values) == t._len "Column length $(length(values)) != table length $(t._len)"
    new_cols = copy(t.columns)
    new_cols[name] = values
    return Table(new_cols)
end
