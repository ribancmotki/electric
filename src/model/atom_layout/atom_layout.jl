"""
atom_layout/atom_layout.jl — AtomLayout, Residues, GatherInfo, and tokenizer.
"""

using Logging

# ──────────────────────────────────────────────────────────────────────────────
# AtomLayout
# ──────────────────────────────────────────────────────────────────────────────

"""
    AtomLayout

Immutable struct-of-arrays describing atom layout.
All non-nothing arrays must have the same shape.
"""
struct AtomLayout
    atom_name::Array{String}
    res_id::Array{Int}
    chain_id::Array{String}
    atom_element::Union{Array{String},Nothing}
    res_name::Union{Array{String},Nothing}
    chain_type::Union{Array{String},Nothing}
end

function AtomLayout(;
    atom_name::Array{String},
    res_id::Array{Int},
    chain_id::Array{String},
    atom_element::Union{Array{String},Nothing} = nothing,
    res_name::Union{Array{String},Nothing} = nothing,
    chain_type::Union{Array{String},Nothing} = nothing,
)
    sh = size(atom_name)
    @assert size(res_id)    == sh "res_id shape mismatch: $(size(res_id)) ≠ $sh"
    @assert size(chain_id)  == sh "chain_id shape mismatch: $(size(chain_id)) ≠ $sh"
    atom_element !== nothing && @assert size(atom_element) == sh "atom_element shape mismatch"
    res_name     !== nothing && @assert size(res_name)     == sh "res_name shape mismatch"
    chain_type   !== nothing && @assert size(chain_type)   == sh "chain_type shape mismatch"
    return AtomLayout(atom_name, res_id, chain_id, atom_element, res_name, chain_type)
end

function Base.show(io::IO, al::AtomLayout)
    print(io, "AtomLayout$(size(al.atom_name))")
end

function layout_shape(al::AtomLayout)::Tuple
    return size(al.atom_name)
end

function Base.length(al::AtomLayout)
    return length(al.atom_name)
end

function Base.:(==)(a::AtomLayout, b::AtomLayout)::Bool
    # Compare only where atom_name is non-empty (valid atoms)
    mask_a = a.atom_name .!= ""
    mask_b = b.atom_name .!= ""
    mask_a != mask_b && return false
    any(mask_a) || return true
    return (a.atom_name[mask_a] == b.atom_name[mask_b] &&
            a.res_id[mask_a]    == b.res_id[mask_b] &&
            a.chain_id[mask_a]  == b.chain_id[mask_b])
end

function Base.getindex(al::AtomLayout, idxs...)
    return AtomLayout(
        atom_name    = al.atom_name[idxs...],
        res_id       = al.res_id[idxs...],
        chain_id     = al.chain_id[idxs...],
        atom_element = al.atom_element !== nothing ? al.atom_element[idxs...] : nothing,
        res_name     = al.res_name     !== nothing ? al.res_name[idxs...]     : nothing,
        chain_type   = al.chain_type   !== nothing ? al.chain_type[idxs...]   : nothing,
    )
end

"""
    copy_and_pad_to(al::AtomLayout, new_shape::Tuple) -> AtomLayout

Pad all arrays to new_shape with empty strings / zeros.
"""
function copy_and_pad_to(al::AtomLayout, new_shape::Tuple)::AtomLayout
    old_sh = layout_shape(al)
    all(new_shape .>= old_sh) || error("Cannot pad to smaller shape: $old_sh → $new_shape")

    function pad_str(arr, fill="")
        out = fill(fill, new_shape)
        idxs = CartesianIndices(old_sh)
        for idx in idxs
            out[idx] = arr[idx]
        end
        return out
    end

    function pad_int(arr, fill=0)
        out = Base.fill(fill, new_shape)
        idxs = CartesianIndices(old_sh)
        for idx in idxs
            out[idx] = arr[idx]
        end
        return out
    end

    return AtomLayout(
        atom_name    = pad_str(al.atom_name),
        res_id       = pad_int(al.res_id),
        chain_id     = pad_str(al.chain_id),
        atom_element = al.atom_element !== nothing ? pad_str(al.atom_element) : nothing,
        res_name     = al.res_name     !== nothing ? pad_str(al.res_name)     : nothing,
        chain_type   = al.chain_type   !== nothing ? pad_str(al.chain_type)   : nothing,
    )
end

function Base.fill(s::String, shape::Tuple)
    arr = Array{String}(undef, shape)
    fill!(arr, s)
    return arr
end

# ──────────────────────────────────────────────────────────────────────────────
# Residues
# ──────────────────────────────────────────────────────────────────────────────

"""
    Residues

Per-residue metadata for atom layout construction.
"""
struct Residues
    res_name::Vector{String}
    res_id::Vector{Int}
    chain_id::Vector{String}
    chain_type::Vector{String}
    is_start_terminus::Vector{Bool}
    is_end_terminus::Vector{Bool}
    deprotonation::Union{Vector{Set{String}},Nothing}
    smiles_string::Union{Vector{Union{String,Nothing}},Nothing}
end

function Base.show(io::IO, r::Residues)
    print(io, "Residues($(length(r.res_name)) residues)")
end

function Base.length(r::Residues)
    return length(r.res_name)
end

# ──────────────────────────────────────────────────────────────────────────────
# GatherInfo
# ──────────────────────────────────────────────────────────────────────────────

"""
    GatherInfo

Index gather information for mapping between two layouts.
"""
struct GatherInfo
    gather_idxs::Array{Int}
    gather_mask::Array{Bool}
    input_shape::Vector{Int}
end

function Base.show(io::IO, g::GatherInfo)
    print(io, "GatherInfo$(size(g.gather_idxs))")
end

function gather_shape(g::GatherInfo)::Tuple
    return size(g.gather_idxs)
end

function Base.getindex(g::GatherInfo, idxs...)
    return GatherInfo(
        g.gather_idxs[idxs...],
        g.gather_mask[idxs...],
        g.input_shape,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# compute_gather_idxs
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_gather_idxs(source_layout::AtomLayout, target_layout::AtomLayout;
                        fill_value::Int=0) -> GatherInfo

Build gather indices from source to target layout.
"""
function compute_gather_idxs(source_layout::AtomLayout, target_layout::AtomLayout;
                              fill_value::Int=0)::GatherInfo
    # Build lookup: (chain_id, res_id, atom_name) → flat index (1-based)
    source_flat = vec(source_layout.atom_name)
    source_res  = vec(source_layout.res_id)
    source_chain = vec(source_layout.chain_id)

    lookup = Dict{Tuple{String,Int,String},Int}()
    for (i, (an, ri, ci)) in enumerate(zip(source_flat, source_res, source_chain))
        isempty(an) && continue
        key = (ci, ri, an)
        haskey(lookup, key) || (lookup[key] = i)
    end

    target_flat  = vec(target_layout.atom_name)
    target_res   = vec(target_layout.res_id)
    target_chain = vec(target_layout.chain_id)
    n_target = length(target_flat)

    gather_idxs = fill(fill_value, length(target_flat))
    gather_mask = falses(length(target_flat))

    for (j, (an, ri, ci)) in enumerate(zip(target_flat, target_res, target_chain))
        isempty(an) && continue
        key = (ci, ri, an)
        idx = get(lookup, key, 0)
        if idx > 0
            gather_idxs[j] = idx
            gather_mask[j] = true
        end
    end

    target_shape = layout_shape(target_layout)
    return GatherInfo(
        reshape(gather_idxs, target_shape),
        reshape(gather_mask, target_shape),
        collect(size(source_layout.atom_name)),
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# convert (gather operation)
# ──────────────────────────────────────────────────────────────────────────────

"""
    atom_layout_convert(gather_info::GatherInfo, arr::AbstractArray;
                        layout_axes::Tuple) -> AbstractArray

Gather `arr` using `gather_info`. `layout_axes` specifies which axes of `arr`
correspond to the source layout axes. Returns zero for invalid entries.
"""
function atom_layout_convert(gather_info::GatherInfo, arr::AbstractArray;
                              layout_axes::Tuple)::AbstractArray
    # Flatten the layout axes of arr
    sh = size(arr)
    n_layout = length(gather_info.input_shape)
    n_axes = ndims(arr)

    # Identify layout dims (using negative indexing: layout_axes=(-2,-1) means last 2)
    normalized_axes = [a < 0 ? n_axes + a + 1 : a for a in layout_axes]
    batch_axes = [i for i in 1:n_axes if i ∉ normalized_axes]
    other_axes = [i for i in 1:n_axes if i ∉ normalized_axes && i ∉ batch_axes]

    # Flatten layout dims and gather
    flat_layout_size = prod(sh[normalized_axes])
    # Build permutation: batch_dims first, then layout_dims, then feature_dims
    perm = vcat(batch_axes, normalized_axes,
                [i for i in 1:n_axes if i ∉ batch_axes && i ∉ normalized_axes])
    arr_perm = permutedims(arr, perm)

    batch_shape  = sh[batch_axes]
    layout_shape_vec = sh[normalized_axes]
    feat_shape   = sh[[i for i in 1:n_axes if i ∉ batch_axes && i ∉ normalized_axes]]

    arr_flat = reshape(arr_perm, batch_shape..., flat_layout_size, feat_shape...)

    # Gather
    flat_idxs = vec(gather_info.gather_idxs)
    flat_mask  = vec(gather_info.gather_mask)
    n_out = length(flat_idxs)

    out_shape = (batch_shape..., size(gather_info.gather_idxs)..., feat_shape...)
    out = zeros(eltype(arr), out_shape)

    # Simple gather implementation
    target_shape = size(gather_info.gather_idxs)
    for j in 1:n_out
        flat_mask[j] || continue
        src_idx = flat_idxs[j]
        (src_idx < 1 || src_idx > flat_layout_size) && continue
        # Copy from arr_flat[batch..., src_idx, feat...] to out[batch..., j, feat...]
        for bidx in CartesianIndices(batch_shape)
            for fidx in CartesianIndices(feat_shape)
                out[bidx, j, fidx] = arr_flat[bidx, src_idx, fidx]
            end
        end
    end

    return reshape(out, out_shape)
end

# ──────────────────────────────────────────────────────────────────────────────
# make_flat_atom_layout
# ──────────────────────────────────────────────────────────────────────────────

"""
    make_flat_atom_layout(residues::Residues, ccd::Ccd;
                          polymer_ligand_bonds=nothing,
                          ligand_ligand_bonds=nothing,
                          with_hydrogens::Bool=false,
                          skip_unk_residues::Bool=true,
                          drop_ligand_leaving_atoms::Bool=false) -> AtomLayout

Build a flat (1D) AtomLayout from a Residues object.
"""
function make_flat_atom_layout(
    residues::Residues,
    ccd::Ccd;
    polymer_ligand_bonds = nothing,
    ligand_ligand_bonds  = nothing,
    with_hydrogens::Bool = false,
    skip_unk_residues::Bool = true,
    drop_ligand_leaving_atoms::Bool = false,
)::AtomLayout
    atom_names  = String[]
    res_ids     = Int[]
    chain_ids   = String[]
    elements    = String[]
    res_names   = String[]
    chain_types = String[]

    bonded_atoms_by_chain = Dict{String,Set{String}}()
    if polymer_ligand_bonds !== nothing
        for (a1, a2) in polymer_ligand_bonds
            cid, _, an = a1
            s = get!(bonded_atoms_by_chain, cid, Set{String}())
            push!(s, an)
        end
    end

    n = length(residues)
    for i in 1:n
        rn = residues.res_name[i]
        rid = residues.res_id[i]
        cid = residues.chain_id[i]
        ct  = residues.chain_type[i]
        is_start = residues.is_start_terminus[i]
        is_end   = residues.is_end_terminus[i]
        deproto  = residues.deprotonation !== nothing ? residues.deprotonation[i] : Set{String}()
        smiles   = residues.smiles_string !== nothing ? residues.smiles_string[i]  : nothing

        # Get atoms for this residue
        atoms = _get_residue_atoms(rn, ct, ccd; with_hydrogens, smiles)

        if skip_unk_residues && rn in UNKNOWN_TYPES && ct in POLYMER_CHAIN_TYPES
            continue
        end

        # Get atoms to drop based on terminus and bonding context
        bonded = get(bonded_atoms_by_chain, cid, Set{String}())
        drop_atoms = get_link_drop_atoms(rn, ct;
            is_start_terminus  = is_start,
            is_end_terminus    = is_end,
            bonded_atoms       = bonded,
            drop_ligand_leaving_atoms = drop_ligand_leaving_atoms,
        )

        for (an, el) in atoms
            an in drop_atoms && continue
            !with_hydrogens && el in ("H","D") && continue
            an in deproto && continue
            push!(atom_names,  an)
            push!(res_ids,     rid)
            push!(chain_ids,   cid)
            push!(elements,    el)
            push!(res_names,   rn)
            push!(chain_types, ct)
        end
    end

    return AtomLayout(
        atom_name  = atom_names,
        res_id     = res_ids,
        chain_id   = chain_ids,
        atom_element = elements,
        res_name   = res_names,
        chain_type = chain_types,
    )
end

function _get_residue_atoms(res_name::String, chain_type::String, ccd::Ccd;
                             with_hydrogens::Bool=false,
                             smiles::Union{String,Nothing}=nothing)::Vector{Tuple{String,String}}
    # Try CCD lookup first
    comp = get(ccd, res_name, nothing)
    if comp !== nothing
        atoms = comp.atoms
        return [(a.atom_id, a.type_symbol) for a in atoms
                if (with_hydrogens || (a.type_symbol != "H" && a.type_symbol != "D"))]
    end

    # Try SMILES
    if smiles !== nothing
        smiles_atoms = smiles_to_atoms(smiles; include_hydrogens=with_hydrogens)
        return [(a["atom_id"], a["type_symbol"]) for a in smiles_atoms]
    end

    # Fallback: use standard atom list for known residues
    if chain_type == PROTEIN_CHAIN
        return [(an, _guess_element(an)) for an in get_standard_atoms(res_name)]
    elseif chain_type in (RNA_CHAIN, DNA_CHAIN)
        return _default_nucleic_atoms(res_name, chain_type)
    end
    return Tuple{String,String}[]
end

function _guess_element(atom_name::String)::String
    isempty(atom_name) && return "C"
    c1 = atom_name[1]
    c1 == 'C' && return "C"
    c1 == 'N' && return "N"
    c1 == 'O' && return "O"
    c1 == 'S' && return "S"
    c1 == 'P' && return "P"
    return string(c1)
end

function _default_nucleic_atoms(res_name::String, chain_type::String)
    backbone = [("P","P"),("OP1","O"),("OP2","O"),("O5'","O"),("C5'","C"),
                ("C4'","C"),("O4'","O"),("C3'","C"),("O3'","O"),("C2'","C"),
                ("O2'","O"),("C1'","C")]
    chain_type == DNA_CHAIN && (backbone = filter(p->p[1]!="O2'", backbone))

    base_atoms = if res_name in ("A","DA")
        [("N9","N"),("C8","C"),("N7","N"),("C5","C"),("C6","C"),("N6","N"),
         ("N1","N"),("C2","C"),("N3","N"),("C4","C")]
    elseif res_name in ("G","DG")
        [("N9","N"),("C8","C"),("N7","N"),("C5","C"),("C6","C"),("O6","O"),
         ("N1","N"),("C2","C"),("N2","N"),("N3","N"),("C4","C")]
    elseif res_name in ("C","DC")
        [("N1","N"),("C2","C"),("O2","O"),("N3","N"),("C4","C"),("N4","N"),
         ("C5","C"),("C6","C")]
    elseif res_name in ("U",)
        [("N1","N"),("C2","C"),("O2","O"),("N3","N"),("C4","C"),("O4","O"),
         ("C5","C"),("C6","C")]
    elseif res_name in ("DT",)
        [("N1","N"),("C2","C"),("O2","O"),("N3","N"),("C4","C"),("O4","O"),
         ("C5","C"),("C7","C"),("C6","C")]
    else
        Tuple{String,String}[]
    end
    return vcat(backbone, base_atoms)
end

# ──────────────────────────────────────────────────────────────────────────────
# get_link_drop_atoms
# ──────────────────────────────────────────────────────────────────────────────

"""
    get_link_drop_atoms(res_name, chain_type; ...) -> Set{String}

Return the set of atoms to drop based on terminus status and bonding context.
"""
function get_link_drop_atoms(
    res_name::String,
    chain_type::String;
    is_start_terminus::Bool,
    is_end_terminus::Bool,
    bonded_atoms::Set{String} = Set{String}(),
    drop_ligand_leaving_atoms::Bool = false,
)::Set{String}
    drop = Set{String}()

    if chain_type == PROTEIN_CHAIN
        !is_start_terminus && push!(drop, "H2", "H3")
        !is_end_terminus   && push!(drop, "OXT", "HXT")
        res_name == "PRO" && !is_start_terminus && push!(drop, "H")
    elseif chain_type in (RNA_CHAIN, DNA_CHAIN)
        !is_start_terminus && push!(drop, "OP3")
    elseif chain_type in LIGAND_CHAIN_TYPES && drop_ligand_leaving_atoms
        # Drop leaving atoms for glycan residues if not bonded
        if res_name in GLYCAN_LINKING_LIGANDS
            "O1" ∉ bonded_atoms && push!(drop, "O1")
        end
    end

    return drop
end

# ──────────────────────────────────────────────────────────────────────────────
# atom_layout_from_structure
# ──────────────────────────────────────────────────────────────────────────────

"""
    atom_layout_from_structure(s::Structure; fix_non_standard_polymer_res=false) -> AtomLayout
"""
function atom_layout_from_structure(s::Structure;
                                     fix_non_standard_polymer_res::Bool=false)::AtomLayout
    n = length(s)
    atom_names  = copy(s.atom_name)
    res_ids     = copy(s.res_id)
    chain_ids   = copy(s.chain_id)
    elements    = copy(s.atom_element)
    res_names   = copy(s.res_name)
    chain_types = copy(s.chain_type)

    if fix_non_standard_polymer_res
        for i in 1:n
            res_names[i] = fix_non_standard_polymer_res(res_names[i], chain_types[i])
        end
    end

    return AtomLayout(
        atom_name  = atom_names,
        res_id     = res_ids,
        chain_id   = chain_ids,
        atom_element = elements,
        res_name   = res_names,
        chain_type = chain_types,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# residues_from_structure
# ──────────────────────────────────────────────────────────────────────────────

"""
    residues_from_structure(s::Structure; ...) -> Residues
"""
function residues_from_structure(s::Structure;
    include_missing_residues::Bool=true,
    fix_non_standard_polymer_res::Bool=false,
)::Residues
    # Collect unique residues in order
    seen = Set{Tuple{String,Int}}()
    res_names_out  = String[]
    res_ids_out    = Int[]
    chain_ids_out  = String[]
    chain_types_out = String[]

    # Build per-chain residue info
    for i in sortperm(s.res_id)
        cid = s.chain_id[i]
        rid = s.res_id[i]
        key = (cid, rid)
        key in seen && continue
        push!(seen, key)
        rn = s.res_name[i]
        ct = s.chain_type[i]
        fix_non_standard_polymer_res && (rn = fix_non_standard_polymer_res(rn, ct))
        push!(res_names_out,  rn)
        push!(res_ids_out,    rid)
        push!(chain_ids_out,  cid)
        push!(chain_types_out, ct)
    end

    n = length(res_names_out)

    # Determine terminus status per chain
    is_start = falses(n)
    is_end   = falses(n)
    chain_first = Dict{String,Int}()
    chain_last  = Dict{String,Int}()
    for i in 1:n
        cid = chain_ids_out[i]
        if !haskey(chain_first, cid)
            chain_first[cid] = i
        end
        chain_last[cid] = i
    end
    for i in 1:n
        cid = chain_ids_out[i]
        is_start[i] = (chain_first[cid] == i)
        is_end[i]   = (chain_last[cid]  == i)
    end

    return Residues(
        res_names_out, res_ids_out, chain_ids_out, chain_types_out,
        is_start, is_end, nothing, nothing,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# guess_deprotonation
# ──────────────────────────────────────────────────────────────────────────────

"""
    guess_deprotonation(residues::Residues) -> Residues

Assume pH 7, guess deprotonation state. For HIS: prefer HE2 form (keep HD1 deprotonated).
"""
function guess_deprotonation(residues::Residues)::Residues
    n = length(residues)
    deprotonation = [Set{String}() for _ in 1:n]

    for i in 1:n
        rn = residues.res_name[i]
        if rn == "HIS"
            push!(deprotonation[i], "HD1")  # default: HE2 tautomer (δ protonated)
        elseif rn == "ASP"
            push!(deprotonation[i], "HD2")
        elseif rn == "GLU"
            push!(deprotonation[i], "HE2")
        end
    end

    return Residues(
        residues.res_name, residues.res_id, residues.chain_id, residues.chain_type,
        residues.is_start_terminus, residues.is_end_terminus,
        deprotonation, residues.smiles_string,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# fill_in_optional_fields
# ──────────────────────────────────────────────────────────────────────────────

function fill_in_optional_fields(minimal_layout::AtomLayout,
                                  reference_atoms::AtomLayout)::AtomLayout
    sh = layout_shape(minimal_layout)
    fill_str(arr_or_nothing, default) =
        arr_or_nothing === nothing ? fill(default, sh) : arr_or_nothing

    return AtomLayout(
        atom_name   = minimal_layout.atom_name,
        res_id      = minimal_layout.res_id,
        chain_id    = minimal_layout.chain_id,
        atom_element = fill_str(minimal_layout.atom_element, ""),
        res_name    = fill_str(minimal_layout.res_name, ""),
        chain_type  = fill_str(minimal_layout.chain_type, ""),
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# make_structure from flat layout + coords
# ──────────────────────────────────────────────────────────────────────────────

"""
    make_structure_from_layout(flat_layout::AtomLayout, atom_coords::Matrix{Float32},
                                name::String; atom_b_factors=nothing,
                                all_physical_residues=nothing) -> Structure
"""
function make_structure_from_layout(
    flat_layout::AtomLayout,
    atom_coords::Matrix{Float32},
    name::String;
    atom_b_factors::Union{Vector{Float32},Nothing} = nothing,
    all_physical_residues = nothing,
)::Structure
    n = length(flat_layout)
    @assert size(atom_coords, 1) >= n "Need at least $n coordinate rows"

    return Structure(
        atom_name    = flat_layout.atom_name,
        atom_element = flat_layout.atom_element !== nothing ? flat_layout.atom_element : fill("", n),
        res_name     = flat_layout.res_name     !== nothing ? flat_layout.res_name     : fill("", n),
        res_id       = flat_layout.res_id,
        chain_id     = flat_layout.chain_id,
        chain_type   = flat_layout.chain_type   !== nothing ? flat_layout.chain_type   : fill("", n),
        atom_x       = atom_coords[1:n, 1],
        atom_y       = atom_coords[1:n, 2],
        atom_z       = atom_coords[1:n, 3],
        atom_b_factor   = atom_b_factors !== nothing ? atom_b_factors : fill(0f0, n),
        atom_occupancy  = fill(1f0, n),
        name         = name,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# tokenizer
# ──────────────────────────────────────────────────────────────────────────────

"""
    tokenizer(flat_output_layout::AtomLayout, ccd::Ccd,
              max_atoms_per_token::Int=24,
              flatten_non_standard_residues::Bool=true,
              logging_name::String="") -> Tuple{AtomLayout, AtomLayout, Vector{Int32}}

Build token representation from flat atom layout.
Returns (all_tokens, all_token_atoms_layout, standard_token_idxs).
- all_tokens: 1D AtomLayout, one representative atom per token
- all_token_atoms_layout: 2D AtomLayout, shape (num_tokens, max_atoms_per_token)
- standard_token_idxs: indices of standard polymer tokens (for MSA/template cropping)
"""
function tokenizer(
    flat_output_layout::AtomLayout,
    ccd::Ccd,
    max_atoms_per_token::Int = 24,
    flatten_non_standard_residues::Bool = true,
    logging_name::String = "",
)::Tuple{AtomLayout,AtomLayout,Vector{Int32}}
    n_atoms = length(flat_output_layout)
    n_atoms == 0 && return (
        AtomLayout(atom_name=String[], res_id=Int[], chain_id=String[]),
        AtomLayout(atom_name=fill("", (0, max_atoms_per_token)),
                   res_id=zeros(Int, 0, max_atoms_per_token),
                   chain_id=fill("", (0, max_atoms_per_token))),
        Int32[],
    )

    # Group atoms by (chain_id, res_id)
    residue_groups = _group_atoms_by_residue(flat_output_layout)

    tokens_atom_name  = String[]
    tokens_res_id     = Int[]
    tokens_chain_id   = String[]
    tokens_element    = String[]
    tokens_res_name   = String[]
    tokens_chain_type = String[]

    tok_atoms_atom_name  = Vector{Vector{String}}()
    tok_atoms_res_id     = Vector{Vector{Int}}()
    tok_atoms_chain_id   = Vector{Vector{String}}()
    tok_atoms_element    = Vector{Vector{String}}()
    tok_atoms_res_name   = Vector{Vector{String}}()
    tok_atoms_chain_type = Vector{Vector{String}}()

    standard_idxs = Int32[]
    tok_idx = 0

    for (cid, rid, rn, ct) in residue_groups
        # Get all atom indices in this residue
        atom_idxs = _get_residue_atom_indices(flat_output_layout, cid, rid)
        isempty(atom_idxs) && continue

        an_list = flat_output_layout.atom_name[atom_idxs]
        el_list = flat_output_layout.atom_element !== nothing ?
                  flat_output_layout.atom_element[atom_idxs] : fill("", length(atom_idxs))

        # Determine if this is a standard polymer residue or non-polymer
        is_standard_polymer = (ct in STANDARD_POLYMER_CHAIN_TYPES &&
                                rn ∉ UNKNOWN_TYPES)
        is_nonstandard_polymer = (ct in STANDARD_POLYMER_CHAIN_TYPES &&
                                   rn in UNKNOWN_TYPES)
        is_ligand = ct in LIGAND_CHAIN_TYPES

        if is_standard_polymer
            # One token: representative atom
            rep_atom = _representative_atom(rn, ct, an_list)
            tok_idx += 1
            push!(standard_idxs, Int32(tok_idx))

            push!(tokens_atom_name, rep_atom)
            push!(tokens_res_id,    rid)
            push!(tokens_chain_id,  cid)
            push!(tokens_element,   _get_element(rep_atom, el_list, an_list))
            push!(tokens_res_name,  rn)
            push!(tokens_chain_type, ct)

            # Token atoms: up to max_atoms_per_token from CCD
            ccd_atoms = _get_ccd_atom_names(rn, ccd; max_atoms=max_atoms_per_token)
            push!(tok_atoms_atom_name,  _pad_to(ccd_atoms, max_atoms_per_token, ""))
            push!(tok_atoms_res_id,     fill(rid, max_atoms_per_token))
            push!(tok_atoms_chain_id,   fill(cid, max_atoms_per_token))
            push!(tok_atoms_element,    fill("", max_atoms_per_token))
            push!(tok_atoms_res_name,   fill(rn,  max_atoms_per_token))
            push!(tok_atoms_chain_type, fill(ct,  max_atoms_per_token))

        elseif is_ligand || (is_nonstandard_polymer && flatten_non_standard_residues)
            # One token per atom
            for (ai, an) in enumerate(an_list)
                tok_idx += 1
                el = el_list[ai]
                push!(tokens_atom_name, an)
                push!(tokens_res_id,    rid)
                push!(tokens_chain_id,  cid)
                push!(tokens_element,   el)
                push!(tokens_res_name,  rn)
                push!(tokens_chain_type, ct)

                # Token atoms: single atom (pad rest)
                t_atoms = fill("", max_atoms_per_token)
                t_atoms[1] = an
                push!(tok_atoms_atom_name,  t_atoms)
                push!(tok_atoms_res_id,     fill(rid, max_atoms_per_token))
                push!(tok_atoms_chain_id,   fill(cid, max_atoms_per_token))
                push!(tok_atoms_element,    fill(el,  max_atoms_per_token))
                push!(tok_atoms_res_name,   fill(rn,  max_atoms_per_token))
                push!(tok_atoms_chain_type, fill(ct,  max_atoms_per_token))
            end
        end
    end

    num_tokens = length(tokens_atom_name)
    num_tokens == 0 && return (
        AtomLayout(atom_name=String[], res_id=Int[], chain_id=String[]),
        AtomLayout(atom_name=fill("", (0, max_atoms_per_token)),
                   res_id=zeros(Int, 0, max_atoms_per_token),
                   chain_id=fill("", (0, max_atoms_per_token))),
        Int32[],
    )

    all_tokens = AtomLayout(
        atom_name  = tokens_atom_name,
        res_id     = tokens_res_id,
        chain_id   = tokens_chain_id,
        atom_element = tokens_element,
        res_name   = tokens_res_name,
        chain_type = tokens_chain_type,
    )

    # Build 2D token-atom layout
    ta_atom_name  = Matrix{String}(undef, num_tokens, max_atoms_per_token)
    ta_res_id     = Matrix{Int}(undef, num_tokens, max_atoms_per_token)
    ta_chain_id   = Matrix{String}(undef, num_tokens, max_atoms_per_token)
    ta_element    = Matrix{String}(undef, num_tokens, max_atoms_per_token)
    ta_res_name   = Matrix{String}(undef, num_tokens, max_atoms_per_token)
    ta_chain_type = Matrix{String}(undef, num_tokens, max_atoms_per_token)

    for (i, (an, ri, ci, el, rn_v, ct_v)) in enumerate(zip(
            tok_atoms_atom_name, tok_atoms_res_id, tok_atoms_chain_id,
            tok_atoms_element, tok_atoms_res_name, tok_atoms_chain_type))
        ta_atom_name[i,  :] = an
        ta_res_id[i,     :] = ri
        ta_chain_id[i,   :] = ci
        ta_element[i,    :] = el
        ta_res_name[i,   :] = rn_v
        ta_chain_type[i, :] = ct_v
    end

    all_token_atoms = AtomLayout(
        atom_name  = ta_atom_name,
        res_id     = ta_res_id,
        chain_id   = ta_chain_id,
        atom_element = ta_element,
        res_name   = ta_res_name,
        chain_type = ta_chain_type,
    )

    return (all_tokens, all_token_atoms, standard_idxs)
end

# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer helpers
# ──────────────────────────────────────────────────────────────────────────────

function _group_atoms_by_residue(al::AtomLayout)
    seen = Set{Tuple{String,Int,String,String}}()
    groups = Tuple{String,Int,String,String}[]
    n = length(al)
    ct_arr = al.chain_type !== nothing ? al.chain_type : fill("", n)
    rn_arr = al.res_name   !== nothing ? al.res_name   : fill("", n)
    for i in 1:n
        key = (al.chain_id[i], al.res_id[i], rn_arr[i], ct_arr[i])
        key in seen && continue
        push!(seen, key)
        push!(groups, key)
    end
    return groups
end

function _get_residue_atom_indices(al::AtomLayout, cid::String, rid::Int)::Vector{Int}
    return findall(i -> al.chain_id[i] == cid && al.res_id[i] == rid, 1:length(al))
end

function _representative_atom(res_name::String, chain_type::String,
                                atom_names::Vector{String})::String
    if chain_type == PROTEIN_CHAIN
        "CA" in atom_names && return "CA"
    elseif chain_type in (RNA_CHAIN, DNA_CHAIN)
        "C1'" in atom_names && return "C1'"
    end
    isempty(atom_names) && return ""
    return atom_names[1]
end

function _get_element(atom_name::String, el_list::Vector{String},
                       an_list::Vector{String})::String
    idx = findfirst(==(atom_name), an_list)
    idx !== nothing && return el_list[idx]
    return _guess_element(atom_name)
end

function _get_ccd_atom_names(res_name::String, ccd::Ccd;
                              max_atoms::Int=24)::Vector{String}
    comp = get(ccd, res_name, nothing)
    if comp !== nothing
        heavy_atoms = filter(a -> a.type_symbol != "H" && a.type_symbol != "D", comp.atoms)
        names = [a.atom_id for a in heavy_atoms[1:min(max_atoms, length(heavy_atoms))]]
        return names
    end
    # Fallback for standard residues
    names = get_standard_atoms(res_name)
    return names[1:min(max_atoms, length(names))]
end

function _pad_to(v::Vector{String}, n::Int, fill_val::String)::Vector{String}
    result = fill(fill_val, n)
    for (i, x) in enumerate(v)
        i > n && break
        result[i] = x
    end
    return result
end
