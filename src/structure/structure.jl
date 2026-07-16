"""
structure.jl — Structure type and accessor functions.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Structure type
# ──────────────────────────────────────────────────────────────────────────────

"""
    Structure

Represents an atomic structure (protein, RNA, DNA, or ligand complex).
All atom data is stored in parallel arrays.
"""
struct Structure
    # Atom arrays
    atom_name::Vector{String}
    atom_element::Vector{String}
    res_name::Vector{String}
    res_id::Vector{Int}
    chain_id::Vector{String}
    chain_type::Vector{String}
    atom_x::Vector{Float32}
    atom_y::Vector{Float32}
    atom_z::Vector{Float32}
    atom_b_factor::Vector{Float32}
    atom_occupancy::Vector{Float32}

    # Metadata
    name::String
    release_date::Union{String,Nothing}
    chemical_components_data::Any    # Ccd or nothing

    # Missing residue registry: chain_id → [(res_name, res_id), ...]
    all_residues::Dict{String,Vector{Tuple{String,Int}}}
end

function Structure(;
    atom_name::Vector{String}    = String[],
    atom_element::Vector{String} = String[],
    res_name::Vector{String}     = String[],
    res_id::Vector{Int}          = Int[],
    chain_id::Vector{String}     = String[],
    chain_type::Vector{String}   = String[],
    atom_x::Vector{Float32}      = Float32[],
    atom_y::Vector{Float32}      = Float32[],
    atom_z::Vector{Float32}      = Float32[],
    atom_b_factor::Vector{Float32}   = Float32[],
    atom_occupancy::Vector{Float32}  = Float32[],
    name::String = "",
    release_date::Union{String,Nothing} = nothing,
    chemical_components_data = nothing,
    all_residues::Dict{String,Vector{Tuple{String,Int}}} = Dict{String,Vector{Tuple{String,Int}}}(),
)
    n = length(atom_name)
    atom_element = isempty(atom_element) ? fill("", n) : atom_element
    atom_b_factor  = isempty(atom_b_factor)  ? fill(0f0, n) : atom_b_factor
    atom_occupancy = isempty(atom_occupancy) ? fill(1f0, n) : atom_occupancy
    return Structure(
        atom_name, atom_element, res_name, res_id, chain_id, chain_type,
        atom_x, atom_y, atom_z, atom_b_factor, atom_occupancy,
        name, release_date, chemical_components_data, all_residues,
    )
end

Base.length(s::Structure) = length(s.atom_name)

function Base.show(io::IO, s::Structure)
    print(io, "Structure(\"$(s.name)\", $(length(s)) atoms, chains=$(join(chains(s),',')))")
end

# ──────────────────────────────────────────────────────────────────────────────
# Accessors
# ──────────────────────────────────────────────────────────────────────────────

"""
    chains(s::Structure) -> Vector{String}

Unique chain IDs in order of first appearance.
"""
function chains(s::Structure)::Vector{String}
    seen = Set{String}()
    result = String[]
    for cid in s.chain_id
        if cid ∉ seen
            push!(seen, cid)
            push!(result, cid)
        end
    end
    return result
end

"""
    coords(s::Structure) -> Matrix{Float32}

Return (N_atoms, 3) coordinate matrix.
"""
function coords(s::Structure)::Matrix{Float32}
    n = length(s)
    m = Matrix{Float32}(undef, n, 3)
    m[:, 1] = s.atom_x
    m[:, 2] = s.atom_y
    m[:, 3] = s.atom_z
    return m
end

"""
    filter(s::Structure; chain_id=nothing, res_id=nothing, atom_name=nothing,
           chain_type=nothing, res_name=nothing) -> Structure

Return a new Structure with only atoms matching the given criteria.
"""
function Base.filter(s::Structure;
    chain_id::Union{String,Vector{String},Nothing} = nothing,
    res_id::Union{Int,Vector{Int},Nothing} = nothing,
    atom_name::Union{String,Vector{String},Nothing} = nothing,
    chain_type::Union{String,Vector{String},Nothing} = nothing,
    res_name::Union{String,Vector{String},Nothing} = nothing,
)::Structure
    mask = trues(length(s))
    if chain_id !== nothing
        ids = chain_id isa String ? Set([chain_id]) : Set(chain_id)
        mask .&= in.(s.chain_id, Ref(ids))
    end
    if res_id !== nothing
        rids = res_id isa Int ? Set([res_id]) : Set(res_id)
        mask .&= in.(s.res_id, Ref(rids))
    end
    if atom_name !== nothing
        names = atom_name isa String ? Set([atom_name]) : Set(atom_name)
        mask .&= in.(s.atom_name, Ref(names))
    end
    if chain_type !== nothing
        types = chain_type isa String ? Set([chain_type]) : Set(chain_type)
        mask .&= in.(s.chain_type, Ref(types))
    end
    if res_name !== nothing
        rnames = res_name isa String ? Set([res_name]) : Set(res_name)
        mask .&= in.(s.res_name, Ref(rnames))
    end
    return _subset(s, mask)
end

"""
    filter_to_entity_type(s::Structure; protein=false, rna=false, dna=false) -> Structure

Filter structure to specified entity types.
"""
function filter_to_entity_type(s::Structure;
    protein::Bool=false, rna::Bool=false, dna::Bool=false)::Structure
    types = String[]
    protein && push!(types, PROTEIN_CHAIN)
    rna     && push!(types, RNA_CHAIN)
    dna     && push!(types, DNA_CHAIN)
    isempty(types) && return s
    return Base.filter(s; chain_type=types)
end

function _subset(s::Structure, mask::Union{BitVector,Vector{Bool}})::Structure
    idxs = findall(mask)
    return Structure(
        atom_name    = s.atom_name[idxs],
        atom_element = s.atom_element[idxs],
        res_name     = s.res_name[idxs],
        res_id       = s.res_id[idxs],
        chain_id     = s.chain_id[idxs],
        chain_type   = s.chain_type[idxs],
        atom_x       = s.atom_x[idxs],
        atom_y       = s.atom_y[idxs],
        atom_z       = s.atom_z[idxs],
        atom_b_factor   = s.atom_b_factor[idxs],
        atom_occupancy  = s.atom_occupancy[idxs],
        name         = s.name,
        release_date = s.release_date,
        chemical_components_data = s.chemical_components_data,
        all_residues = s.all_residues,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Sequence accessors
# ──────────────────────────────────────────────────────────────────────────────

"""
    chain_single_letter_sequence(s::Structure) -> Dict{String,String}

Per-chain single-letter amino acid / nucleotide sequence.
"""
function chain_single_letter_sequence(s::Structure)::Dict{String,String}
    result = Dict{String,String}()
    for cid in chains(s)
        chain_mask = s.chain_id .== cid
        ct = s.chain_type[findfirst(chain_mask)]
        # Collect unique (res_id, res_name) in order
        seen_res = Set{Int}()
        seq_chars = Char[]
        res_ids   = s.res_id[chain_mask]
        res_names = s.res_name[chain_mask]
        perm = sortperm(res_ids)
        for i in perm
            rid = res_ids[i]
            rid in seen_res && continue
            push!(seen_res, rid)
            rn = res_names[i]
            one_letter = letters_three_to_one(rn; default="X")
            push!(seq_chars, only(one_letter))
        end
        result[cid] = String(seq_chars)
    end
    return result
end

"""
    chain_res_name_sequence(s::Structure; include_missing_residues=true,
                            fix_non_standard_polymer_res=false) -> Dict{String,Vector{String}}

Per-chain vector of residue names (CCD codes).
"""
function chain_res_name_sequence(s::Structure;
    include_missing_residues::Bool=true,
    fix_non_standard_polymer_res::Bool=false,
)::Dict{String,Vector{String}}
    result = Dict{String,Vector{String}}()
    for cid in chains(s)
        chain_mask = s.chain_id .== cid
        ct = s.chain_type[findfirst(chain_mask)]
        seen_res = Set{Int}()
        res_list = Tuple{Int,String}[]
        res_ids   = s.res_id[chain_mask]
        res_names = s.res_name[chain_mask]
        for i in sortperm(res_ids)
            rid = res_ids[i]
            rid in seen_res && continue
            push!(seen_res, rid)
            rn = res_names[i]
            if fix_non_standard_polymer_res
                rn = fix_non_standard_polymer_res(rn, ct)
            end
            push!(res_list, (rid, rn))
        end
        # Optionally include missing residues
        if include_missing_residues && haskey(s.all_residues, cid)
            for (rn, rid) in s.all_residues[cid]
                rid ∉ seen_res && push!(res_list, (rid, rn))
            end
            sort!(res_list, by=x->x[1])
        end
        result[cid] = [rn for (_, rn) in res_list]
    end
    return result
end

# ──────────────────────────────────────────────────────────────────────────────
# Iterators
# ──────────────────────────────────────────────────────────────────────────────

"""
    iter_chains(s::Structure)

Iterate over unique chains, yielding Dict with keys "chain_id", "chain_type".
"""
function iter_chains(s::Structure)
    result = Dict{String,Any}[]
    for cid in chains(s)
        mask = s.chain_id .== cid
        ct = s.chain_type[findfirst(mask)]
        push!(result, Dict("chain_id"=>cid, "chain_type"=>ct))
    end
    return result
end

"""
    iter_residues(s::Structure)

Iterate over unique (chain_id, res_id) pairs.
"""
function iter_residues(s::Structure)
    result = Dict{String,Any}[]
    seen = Set{Tuple{String,Int}}()
    for i in 1:length(s)
        key = (s.chain_id[i], s.res_id[i])
        key in seen && continue
        push!(seen, key)
        push!(result, Dict(
            "chain_id" => s.chain_id[i],
            "chain_type" => s.chain_type[i],
            "res_id"   => s.res_id[i],
            "res_name" => s.res_name[i],
        ))
    end
    return result
end

"""
    iter_atoms(s::Structure)

Iterate over all atoms as Dicts.
"""
function iter_atoms(s::Structure)
    n = length(s)
    result = Vector{Dict{String,Any}}(undef, n)
    for i in 1:n
        result[i] = Dict{String,Any}(
            "atom_name"    => s.atom_name[i],
            "atom_element" => s.atom_element[i],
            "res_name"     => s.res_name[i],
            "res_id"       => s.res_id[i],
            "chain_id"     => s.chain_id[i],
            "chain_type"   => s.chain_type[i],
            "x"            => s.atom_x[i],
            "y"            => s.atom_y[i],
            "z"            => s.atom_z[i],
            "b_factor"     => s.atom_b_factor[i],
            "occupancy"    => s.atom_occupancy[i],
        )
    end
    return result
end

# ──────────────────────────────────────────────────────────────────────────────
# Copy and update
# ──────────────────────────────────────────────────────────────────────────────

function copy_and_update_atoms(s::Structure;
    atom_x::Union{Vector{Float32},Nothing} = nothing,
    atom_y::Union{Vector{Float32},Nothing} = nothing,
    atom_z::Union{Vector{Float32},Nothing} = nothing,
    atom_b_factor::Union{Vector{Float32},Nothing} = nothing,
    atom_occupancy::Union{Vector{Float32},Nothing} = nothing,
)::Structure
    return Structure(
        atom_name    = s.atom_name,
        atom_element = s.atom_element,
        res_name     = s.res_name,
        res_id       = s.res_id,
        chain_id     = s.chain_id,
        chain_type   = s.chain_type,
        atom_x       = atom_x       === nothing ? s.atom_x       : atom_x,
        atom_y       = atom_y       === nothing ? s.atom_y       : atom_y,
        atom_z       = atom_z       === nothing ? s.atom_z       : atom_z,
        atom_b_factor   = atom_b_factor   === nothing ? s.atom_b_factor   : atom_b_factor,
        atom_occupancy  = atom_occupancy  === nothing ? s.atom_occupancy  : atom_occupancy,
        name         = s.name,
        release_date = s.release_date,
        chemical_components_data = s.chemical_components_data,
        all_residues = s.all_residues,
    )
end

function copy_and_update_globals(s::Structure;
    release_date::Union{String,Nothing} = s.release_date,
    name::Union{String,Nothing} = nothing,
)::Structure
    return Structure(
        atom_name    = s.atom_name,
        atom_element = s.atom_element,
        res_name     = s.res_name,
        res_id       = s.res_id,
        chain_id     = s.chain_id,
        chain_type   = s.chain_type,
        atom_x       = s.atom_x,
        atom_y       = s.atom_y,
        atom_z       = s.atom_z,
        atom_b_factor   = s.atom_b_factor,
        atom_occupancy  = s.atom_occupancy,
        name         = name === nothing ? s.name : name,
        release_date = release_date,
        chemical_components_data = s.chemical_components_data,
        all_residues = s.all_residues,
    )
end

function rename_chain_ids(s::Structure; new_id_by_old_id::Dict{String,String})::Structure
    new_chain_id = [get(new_id_by_old_id, cid, cid) for cid in s.chain_id]
    new_all_res = Dict{String,Vector{Tuple{String,Int}}}(
        get(new_id_by_old_id, k, k) => v for (k,v) in s.all_residues
    )
    return Structure(
        atom_name    = s.atom_name,
        atom_element = s.atom_element,
        res_name     = s.res_name,
        res_id       = s.res_id,
        chain_id     = new_chain_id,
        chain_type   = s.chain_type,
        atom_x       = s.atom_x,
        atom_y       = s.atom_y,
        atom_z       = s.atom_z,
        atom_b_factor   = s.atom_b_factor,
        atom_occupancy  = s.atom_occupancy,
        name         = s.name,
        release_date = s.release_date,
        chemical_components_data = s.chemical_components_data,
        all_residues = new_all_res,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Multi-sample split
# ──────────────────────────────────────────────────────────────────────────────

"""
    unstack(s::Structure) -> Vector{Structure}

Split a multi-model structure into individual structures.
For now, returns a single-element vector (placeholder for future model stacking).
"""
function unstack(s::Structure)::Vector{Structure}
    return [s]
end

# ──────────────────────────────────────────────────────────────────────────────
# Construction from arrays
# ──────────────────────────────────────────────────────────────────────────────

"""
    structure_from_arrays(; kwargs...) -> Structure

Construct a Structure from parallel arrays. Alias for Structure constructor.
"""
structure_from_arrays(; kwargs...) = Structure(; kwargs...)
