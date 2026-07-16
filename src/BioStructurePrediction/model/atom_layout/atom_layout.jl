"""
Atom layout utilities: mapping between token-level and per-atom representations.
"""

# Re-export the standard atom mask and order from atom_types.jl
# (STANDARD_ATOM_MASK and ATOM_ORDER are defined there)

"""
    atom_layout_to_flat(
        positions::AbstractArray{Float32,3},
        mask::AbstractMatrix{Bool}
    ) -> Tuple{Matrix{Float32}, Vector{Bool}}

Flatten a dense (num_tokens, NUM_ATOM_SLOTS, 3) position array and
corresponding (num_tokens, NUM_ATOM_SLOTS) mask to flat arrays:
- flat_positions: (num_atoms, 3) — only present atoms (mask=true)
- flat_mask: (num_atoms,) — all true (since we filter to present only)

Returns (flat_positions, flat_mask).
"""
function atom_layout_to_flat(
    positions::AbstractArray{Float32,3},
    mask::AbstractMatrix{Bool},
)::Tuple{Matrix{Float32}, Vector{Bool}}
    num_tokens, num_slots, _ = size(positions)
    size(mask) == (num_tokens, num_slots) ||
        error("positions shape $(size(positions)) incompatible with mask shape $(size(mask))")

    flat_pos  = Float32[]
    flat_mask = Bool[]

    for i in 1:num_tokens
        for j in 1:num_slots
            if mask[i, j]
                push!(flat_pos, positions[i, j, 1])
                push!(flat_pos, positions[i, j, 2])
                push!(flat_pos, positions[i, j, 3])
                push!(flat_mask, true)
            end
        end
    end

    n_atoms = length(flat_mask)
    return reshape(Float32.(flat_pos), n_atoms, 3), flat_mask
end

"""
    flat_to_atom_layout(
        flat_positions::AbstractMatrix{Float32},
        token_atom_counts::AbstractVector{Int}
    ) -> Array{Float32,3}

Invert atom_layout_to_flat: given flat positions and per-token atom counts,
reconstruct the dense (num_tokens, max_atoms, 3) array.

token_atom_counts: number of atoms for each token
"""
function flat_to_atom_layout(
    flat_positions::AbstractMatrix{Float32},
    token_atom_counts::AbstractVector{Int},
)::Array{Float32,3}
    n_tokens  = length(token_atom_counts)
    max_atoms = maximum(token_atom_counts; init=0)
    max_atoms == 0 && return zeros(Float32, n_tokens, 0, 3)

    dense = zeros(Float32, n_tokens, max_atoms, 3)
    offset = 0
    for (i, count) in enumerate(token_atom_counts)
        for j in 1:count
            flat_idx = offset + j
            flat_idx <= size(flat_positions, 1) || break
            dense[i, j, :] = flat_positions[flat_idx, :]
        end
        offset += count
    end
    return dense
end

"""
    get_token_atom_mask(
        token_residue_types::Vector{String},
        ccd::Union{Ccd,Nothing} = nothing
    ) -> Matrix{Bool}

Get the per-token atom presence mask of shape (num_tokens, NUM_ATOM_SLOTS).
For polymer residues, uses STANDARD_ATOM_MASK.
For ligands, uses CCD atom data if available.
"""
function get_token_atom_mask(
    token_residue_types::Vector{String},
    ccd::Union{Ccd,Nothing} = nothing,
)::Matrix{Bool}
    n = length(token_residue_types)
    mask = falses(n, NUM_ATOM_SLOTS)
    for (i, res_type) in enumerate(token_residue_types)
        std_mask = get(STANDARD_ATOM_MASK, res_type, nothing)
        if std_mask !== nothing
            for j in 1:min(length(std_mask), NUM_ATOM_SLOTS)
                mask[i, j] = std_mask[j]
            end
        else
            # Unknown type: mark only first atom
            mask[i, 1] = true
        end
    end
    return mask
end

"""
    dense_positions_from_structure(
        s::Structure,
        token_residue_types::Vector{String},
        token_chain_ids::Vector{String},
        token_seq_ids::Vector{String},
    ) -> Tuple{Array{Float32,3}, Matrix{Bool}}

Extract dense (num_tokens, NUM_ATOM_SLOTS, 3) positions and mask from a Structure.

Returns:
- positions: Float32 (num_tokens, NUM_ATOM_SLOTS, 3), zeros for absent atoms
- mask: Bool (num_tokens, NUM_ATOM_SLOTS), true for present atoms
"""
function dense_positions_from_structure(
    s::Structure,
    token_residue_types::Vector{String},
    token_chain_ids::Vector{String},
    token_seq_ids::Vector{String},
)::Tuple{Array{Float32,3}, Matrix{Bool}}
    n = length(token_residue_types)
    positions = zeros(Float32, n, NUM_ATOM_SLOTS, 3)
    mask      = falses(n, NUM_ATOM_SLOTS)

    # Build atom lookup: (chain_id, seq_id, atom_name) → position
    s_chain_ids  = get_column(s.atoms, :label_asym_id)
    s_seq_ids    = get_column(s.atoms, :label_seq_id)
    s_atom_names = get_column(s.atoms, :label_atom_id)
    s_xs         = get_column(s.atoms, :Cartn_x)
    s_ys         = get_column(s.atoms, :Cartn_y)
    s_zs         = get_column(s.atoms, :Cartn_z)

    atom_index = Dict{Tuple{String,String,String},Tuple{Float32,Float32,Float32}}()
    for i in 1:num_atoms(s)
        key = (s_chain_ids[i], string(s_seq_ids[i]), s_atom_names[i])
        atom_index[key] = (s_xs[i], s_ys[i], s_zs[i])
    end

    for (ti, (res_type, cid, sid)) in enumerate(zip(token_residue_types, token_chain_ids, token_seq_ids))
        atom_order_for_res = get(ATOM_ORDER, res_type, String[])
        for (j, atom_name) in enumerate(atom_order_for_res)
            j > NUM_ATOM_SLOTS && break
            isempty(atom_name) && continue
            key = (cid, sid, atom_name)
            pos = get(atom_index, key, nothing)
            if pos !== nothing
                positions[ti, j, 1] = pos[1]
                positions[ti, j, 2] = pos[2]
                positions[ti, j, 3] = pos[3]
                mask[ti, j] = true
            end
        end
    end

    return positions, mask
end

"""
    build_structure_from_dense_positions(;
        positions, mask, token_residue_types, token_chain_ids, token_seq_ids,
        bfactors, name
    ) -> Structure

Reconstruct a Structure from dense position arrays.
"""
function build_structure_from_dense_positions(;
    positions::AbstractArray{Float32,3},
    mask::AbstractMatrix{Bool},
    token_residue_types::AbstractVector{String},
    token_chain_ids::AbstractVector{String},
    token_seq_ids::AbstractVector{String},
    bfactors::AbstractVector{Float32},
    name::String = "predicted",
)::Structure
    n_tokens = length(token_residue_types)
    all_chain_ids  = String[]
    all_res_ids    = String[]
    all_comp_ids   = String[]
    all_atom_names = String[]
    all_elements   = String[]
    all_positions  = Float32[]
    all_bfactors   = Float32[]

    for ti in 1:n_tokens
        res_type  = token_residue_types[ti]
        cid       = token_chain_ids[ti]
        sid       = token_seq_ids[ti]
        atom_list = get(ATOM_ORDER, res_type, String[])
        bf        = ti <= length(bfactors) ? bfactors[ti] : 0f0

        for (j, atom_name) in enumerate(atom_list)
            j > NUM_ATOM_SLOTS && break
            isempty(atom_name) && continue
            mask[ti, j] || continue

            push!(all_chain_ids,  cid)
            push!(all_res_ids,    sid)
            push!(all_comp_ids,   res_type)
            push!(all_atom_names, atom_name)

            # Determine element from atom name
            elem = first_element_char(atom_name)
            push!(all_elements,   elem)

            push!(all_positions, positions[ti, j, 1])
            push!(all_positions, positions[ti, j, 2])
            push!(all_positions, positions[ti, j, 3])
            push!(all_bfactors,  bf)
        end
    end

    n_atoms = length(all_bfactors)
    n_atoms == 0 && return Structure(name, StructureTable(), ChainInfo[], ResidueInfo[], Bond[])

    pos_mat = reshape(Float32.(all_positions), n_atoms, 3)
    return structure_from_arrays(;
        name       = name,
        chain_ids  = all_chain_ids,
        res_ids    = all_res_ids,
        comp_ids   = all_comp_ids,
        atom_names = all_atom_names,
        elements   = all_elements,
        positions  = pos_mat,
        bfactors   = all_bfactors,
    )
end

"""
    first_element_char(atom_name::String) -> String

Guess the element symbol from an atom name.
"""
function first_element_char(atom_name::String)::String
    isempty(atom_name) && return "C"
    # Strip leading digits
    stripped = lstrip(atom_name, ['0':'9'...])
    isempty(stripped) && return "C"
    c = first(stripped)
    # Common multi-char elements
    if length(stripped) >= 2
        two = string(stripped[1], stripped[2])
        two ∈ ("CL","BR","SE","FE","CO","CU","ZN","MG","MN","CA","NA","AL","SI","AS") && return uppercasefirst(lowercase(two))
    end
    return string(c)
end
