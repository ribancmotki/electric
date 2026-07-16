"""
model/components/utils.jl — Shared model utility functions.
"""

using Statistics

"""
    mask_mean(mask, value; axis=nothing, keepdims=false, eps=1f-10)

Compute mean of `value` only over positions where `mask > 0`.
"""
function mask_mean(mask::AbstractArray, value::AbstractArray;
                   axis=nothing, keepdims::Bool=false, eps::Float32=1f-10)
    masked_value = mask .* value
    if axis === nothing
        s = sum(masked_value)
        n = sum(mask) + eps
        return s / n
    else
        s = sum(masked_value, dims=axis)
        n = sum(mask, dims=axis) .+ eps
        result = s ./ n
        keepdims ? result : dropdims(result, dims=axis)
    end
end

"""
    softmax_last_dim(x::AbstractArray) -> AbstractArray

Apply softmax over the last dimension.
"""
function softmax_last_dim(x::AbstractArray)
    x_max = maximum(x, dims=ndims(x))
    ex = exp.(x .- x_max)
    return ex ./ sum(ex, dims=ndims(x))
end

"""
    remove_invalidly_typed_feats(batch::Dict) -> Dict

Keep only arrays with dtypes in {Float32, Float64, Int8, Int32, Int64, Bool}.
"""
function remove_invalidly_typed_feats(batch::Dict)::Dict
    valid_types = Set([Float32, Float64, Int8, Int16, Int32, Int64, Bool, UInt8])
    result = Dict{String,AbstractArray}()
    for (k, v) in batch
        !(v isa AbstractArray) && continue
        eltype(v) in valid_types || continue
        result[k] = v
    end
    return result
end

"""
    bfloat16_context()

No-op on CPU; placeholder for future GPU bfloat16 casting.
Returns a function that casts Float32 to Float32 (identity on CPU).
"""
function bfloat16_context()
    return identity
end

"""
    one_hot(x::AbstractArray{<:Integer}, n_classes::Int) -> AbstractArray{Float32}

One-hot encode integer array. Output has an extra last dimension of size n_classes.
"""
function one_hot(x::AbstractArray{<:Integer}, n_classes::Int)::Array{Float32}
    sh = size(x)
    result = zeros(Float32, sh..., n_classes)
    for idx in CartesianIndices(x)
        v = x[idx]
        (1 <= v <= n_classes) && (result[idx, v] = 1f0)
    end
    return result
end

"""
    create_relative_encoding(token_features, max_relative_idx::Int=32,
                              max_relative_chain::Int=2) -> Array{Float32}

Compute relative position encoding between all token pairs.
Returns array of shape (num_tokens, num_tokens, num_rel_features).
"""
function create_relative_encoding(token_features::Dict, max_relative_idx::Int=32,
                                   max_relative_chain::Int=2)::Array{Float32}
    aatype = get(token_features, "aatype", nothing)
    aatype === nothing && return zeros(Float32, 0, 0, 0)

    n = length(aatype)
    asym_id   = get(token_features, "asym_id",  zeros(Int32, n))
    entity_id = get(token_features, "entity_id", zeros(Int32, n))

    # Residue index relative encoding
    res_idx = get(token_features, "residue_index", Int32.(1:n))

    n_bins = 2 * max_relative_idx + 2  # [-max, max, clipped_far]
    features = zeros(Float32, n, n, n_bins + 2 + 2 * max_relative_chain + 1)

    for i in 1:n, j in 1:n
        feat_idx = 0

        # Residue relative position
        same_chain = (asym_id[i] == asym_id[j])
        if same_chain
            d = Int(res_idx[j]) - Int(res_idx[i])
            d_clipped = clamp(d, -max_relative_idx, max_relative_idx)
            bin_idx = d_clipped + max_relative_idx + 1
            features[i, j, bin_idx] = 1f0
        else
            features[i, j, n_bins] = 1f0  # different chain
        end

        # Same entity indicator
        features[i, j, n_bins+1] = Float32(entity_id[i] == entity_id[j])
    end

    return features
end

"""
    create_msa_feat(msa_batch::Dict) -> Array{Float32}

Build MSA feature tensor from batch.
"""
function create_msa_feat(msa_batch::Dict)::Array{Float32}
    msa_oh = get(msa_batch, "msa_one_hot", nothing)
    msa_oh === nothing && return zeros(Float32, 0, 0, 0)
    has_del = get(msa_batch, "has_deletion",   zeros(Float32, size(msa_oh)[1:2]...))
    del_val = get(msa_batch, "deletion_value", zeros(Float32, size(msa_oh)[1:2]...))
    return cat(msa_oh, has_del[:,:,newaxis], del_val[:,:,newaxis]; dims=3)
end

const newaxis = [CartesianIndex()]

"""
    create_target_feat(batch::Dict; append_per_atom_features::Bool=false) -> Array{Float32}

Build target feature tensor. Shape: (num_tokens, TARGET_FEAT_DIM).
"""
function create_target_feat(batch::Dict;
                             append_per_atom_features::Bool=false)::Array{Float32}
    tf = get(batch, "token_features", nothing)
    tf === nothing && return zeros(Float32, 0, 0)

    aatype   = get(tf, "aatype",   nothing)
    aatype === nothing && return zeros(Float32, 0, 0)

    n = length(aatype)
    n_types = POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP

    # One-hot encode aatype
    aatype_oh = one_hot(Int32.(aatype), n_types)  # (n, n_types)

    # Entity type flags
    is_protein  = Float32.(get(tf, "is_protein",  zeros(Bool, n)))
    is_rna      = Float32.(get(tf, "is_rna",      zeros(Bool, n)))
    is_dna      = Float32.(get(tf, "is_dna",      zeros(Bool, n)))
    is_ligand   = Float32.(get(tf, "is_ligand",   zeros(Bool, n)))

    target_feat = cat(aatype_oh,
                      reshape(is_protein, n, 1),
                      reshape(is_rna,     n, 1),
                      reshape(is_dna,     n, 1),
                      reshape(is_ligand,  n, 1);
                      dims=2)

    return target_feat
end

"""
    dot_product_attention(q, k, v, mask; bias=nothing, implementation=:default)

Multi-head attention via scaled dot product.
"""
function dot_product_attention(q::AbstractArray, k::AbstractArray, v::AbstractArray,
                                mask::Union{AbstractArray,Nothing}=nothing;
                                bias::Union{AbstractArray,Nothing}=nothing,
                                implementation::Symbol=:default)::AbstractArray
    # q: (..., seq_q, heads, head_dim)
    # k: (..., seq_k, heads, head_dim)
    # v: (..., seq_k, heads, head_dim)
    head_dim = size(q, ndims(q))
    scale = Float32(1f0 / sqrt(Float32(head_dim)))

    # logits: (..., heads, seq_q, seq_k)
    # einsum("...qhc,...khc->...hqk", q, k)
    q_perm = permutedims(q, (1:ndims(q)-3..., ndims(q)-1, ndims(q)-2, ndims(q)))
    k_perm = permutedims(k, (1:ndims(k)-3..., ndims(k)-1, ndims(k)-2, ndims(k)))
    logits = batched_mul(q_perm, permutedims(k_perm, (1:ndims(k_perm)-2..., ndims(k_perm), ndims(k_perm)-1))) .* scale

    # Add bias
    bias !== nothing && (logits = logits .+ bias)

    # Add mask
    if mask !== nothing
        # mask: (..., seq) → (..., 1, 1, seq)
        mask_expanded = reshape(mask, size(mask)[1:end-1]..., 1, 1, size(mask, ndims(mask)))
        logits = logits .+ 1f9 .* (mask_expanded .- 1f0)
    end

    weights = softmax_last_dim(logits)

    # weighted_avg: einsum("...hqk,...khc->...qhc", weights, v)
    v_perm = permutedims(v, (1:ndims(v)-3..., ndims(v)-1, ndims(v)-2, ndims(v)))
    wa = batched_mul(weights, v_perm)  # (..., heads, seq_q, head_dim)
    return permutedims(wa, (1:ndims(wa)-3..., ndims(wa)-1, ndims(wa)-3, ndims(wa)))
end

"""
    batched_mul(a, b)

Batched matrix multiply of last two dims.
"""
function batched_mul(a::AbstractArray, b::AbstractArray)
    # (..., m, k) × (..., k, n) → (..., m, n)
    sh_a = size(a)
    sh_b = size(b)
    m, k  = sh_a[end-1], sh_a[end]
    k2, n = sh_b[end-1], sh_b[end]
    @assert k == k2 "batched_mul: inner dims mismatch $k ≠ $k2"

    batch = sh_a[1:end-2]
    a_flat = reshape(a, :, m, k)
    b_flat = reshape(b, :, k, n)
    nb = size(a_flat, 1)
    result = zeros(Float32, nb, m, n)
    for i in 1:nb
        result[i, :, :] = Float32.(a_flat[i,:,:]) * Float32.(b_flat[i,:,:])
    end
    return reshape(result, batch..., m, n)
end
