"""
merging_features.jl — Batch padding, bucket selection, and batch merging.
"""

using Logging

# ──────────────────────────────────────────────────────────────────────────────
# Bucket selection
# ──────────────────────────────────────────────────────────────────────────────

"""
    pad_to_bucket(batch::Dict, bucket_size::Int;
                  token_axis_keys=nothing,
                  msa_token_axis_keys=nothing,
                  pad_value=0) -> Dict

Pad all token-dimension arrays in batch to bucket_size.
"""
function pad_to_bucket(
    batch::Dict,
    bucket_size::Int;
    token_axis_keys::Union{Vector{String},Nothing} = nothing,
    msa_token_axis_keys::Union{Vector{String},Nothing} = nothing,
    pad_value = 0,
)::Dict
    result = Dict{String,Any}()

    for (k, v) in batch
        if !(v isa AbstractArray)
            result[k] = v
            continue
        end

        ndims_v = ndims(v)
        n_toks  = get(batch, "num_tokens", 0)
        n_toks == 0 && (result[k] = v; continue)

        # Detect which axes need padding based on shape matching num_tokens
        sh = size(v)
        padded = _pad_array_to_bucket(v, sh, n_toks, bucket_size, pad_value)
        result[k] = padded
    end

    result["num_tokens"]    = batch["num_tokens"]
    result["bucket_size"]   = bucket_size
    return result
end

function _pad_array_to_bucket(v::AbstractArray, sh::Tuple, n_toks::Int,
                                bucket_size::Int, pad_value)::AbstractArray
    pad_dims = Int[]
    for (di, s) in enumerate(sh)
        s == n_toks && push!(pad_dims, di)
    end
    isempty(pad_dims) && return v

    v_out = copy(v)
    for dim in pad_dims
        n_pad = bucket_size - size(v_out, dim)
        n_pad <= 0 && continue
        pad_sh = collect(size(v_out))
        pad_sh[dim] = n_pad
        pad_arr = fill(eltype(v_out)(pad_value), pad_sh...)
        v_out = cat(v_out, pad_arr; dims=dim)
    end
    return v_out
end

# ──────────────────────────────────────────────────────────────────────────────
# Merge batch dicts
# ──────────────────────────────────────────────────────────────────────────────

"""
    merge_batch_dicts(batches::Vector{Dict}) -> Dict

Merge a list of batch dicts by concatenating arrays along the first dimension.
Used for combining multiple samples/seeds into a single batched tensor.
"""
function merge_batch_dicts(batches::Vector{<:Dict})::Dict
    isempty(batches) && return Dict{String,Any}()
    length(batches) == 1 && return batches[1]

    # Get all keys from first batch
    all_keys = keys(batches[1])
    result = Dict{String,Any}()

    for k in all_keys
        vs = [b[k] for b in batches if haskey(b, k)]
        isempty(vs) && continue

        if all(v isa AbstractArray for v in vs)
            # Stack arrays along a new first dimension
            try
                result[k] = stack_arrays(vs)
            catch e
                @debug "Could not stack $k: $e. Keeping first."
                result[k] = vs[1]
            end
        else
            result[k] = vs[1]
        end
    end

    return result
end

"""
    stack_arrays(arrs::Vector{<:AbstractArray}) -> AbstractArray

Stack arrays along a new axis 1 (batch dimension).
All arrays must have the same shape.
"""
function stack_arrays(arrs::Vector{<:AbstractArray})::AbstractArray
    length(arrs) == 0 && error("Cannot stack empty list")
    length(arrs) == 1 && return reshape(arrs[1], 1, size(arrs[1])...)

    sh = size(arrs[1])
    @assert all(size(a) == sh for a in arrs) "Arrays have different shapes: $(unique(size.(arrs)))"

    T = promote_type(eltype.(arrs)...)
    result = zeros(T, length(arrs), sh...)
    for (i, a) in enumerate(arrs)
        result[i, fill(Colon(), length(sh))...] = a
    end
    return result
end

# ──────────────────────────────────────────────────────────────────────────────
# MSA batching across chains
# ──────────────────────────────────────────────────────────────────────────────

"""
    merge_chain_msas(batch::Dict, num_tokens::Int,
                     max_msa_seqs::Int=512) -> Dict{String,Array}

Gather per-chain MSA features into a single MSA feature block of
shape (max_msa_seqs, num_tokens) padded with zeros.
"""
function merge_chain_msas(batch::Dict, num_tokens::Int;
                           max_msa_seqs::Int=512)::Dict{String,Array}
    msa_parts      = Int8[]
    msa_mask_parts = Bool[]
    del_parts      = Int8[]

    chain_keys = [k for k in keys(batch) if startswith(k, "chain_msa_")]
    for ck in sort(chain_keys)
        cm = batch[ck]
        msa_part  = get(cm, "msa",             nothing)
        mask_part = get(cm, "msa_mask",        nothing)
        del_part  = get(cm, "deletion_matrix", nothing)
        msa_part === nothing && continue

        n_seqs, n_toks = size(msa_part)
        n_toks == num_tokens || continue  # skip mismatched

        isempty(msa_parts) ? (msa_parts = msa_part) : (msa_parts = vcat(msa_parts, msa_part))
        isempty(msa_mask_parts) ? (msa_mask_parts = mask_part) : (msa_mask_parts = vcat(msa_mask_parts, mask_part))
        isempty(del_parts) ? (del_parts = del_part) : (del_parts = vcat(del_parts, del_part))
    end

    if isempty(msa_parts)
        return Dict{String,Array}(
            "msa"             => zeros(Int8, 1, num_tokens),
            "msa_mask"        => ones(Bool,  1, num_tokens),
            "deletion_matrix" => zeros(Int8, 1, num_tokens),
        )
    end

    # Truncate/pad to max_msa_seqs
    n_actual = size(msa_parts, 1)
    if n_actual > max_msa_seqs
        msa_parts      = msa_parts[1:max_msa_seqs, :]
        msa_mask_parts = msa_mask_parts[1:max_msa_seqs, :]
        del_parts      = del_parts[1:max_msa_seqs, :]
    elseif n_actual < max_msa_seqs
        n_pad = max_msa_seqs - n_actual
        msa_parts      = vcat(msa_parts,      zeros(Int8, n_pad, num_tokens))
        msa_mask_parts = vcat(msa_mask_parts,  zeros(Bool, n_pad, num_tokens))
        del_parts      = vcat(del_parts,       zeros(Int8, n_pad, num_tokens))
    end

    return Dict{String,Array}(
        "msa"             => msa_parts,
        "msa_mask"        => msa_mask_parts,
        "deletion_matrix" => del_parts,
    )
end
