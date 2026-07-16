"""
Shared utility functions for model components.
"""

using Flux
using NNlib
using LinearAlgebra

# ──────────────────────────────────────────────
#  LayerNorm
# ──────────────────────────────────────────────

"""
    apply_layer_norm(x, scale, offset) -> Array

Apply layer normalisation: (x - mean) / std * scale + offset.
Normalises over the last dimension.
"""
function apply_layer_norm(x::AbstractArray, scale::AbstractVector, offset::AbstractVector)
    μ  = mean(x; dims=ndims(x))
    σ² = mean((x .- μ).^2; dims=ndims(x))
    σ  = sqrt.(σ² .+ 1f-5)
    return (x .- μ) ./ σ .* reshape(scale, ones(Int, ndims(x)-1)..., :) .+
           reshape(offset, ones(Int, ndims(x)-1)..., :)
end

# ──────────────────────────────────────────────
#  Attention utilities
# ──────────────────────────────────────────────

"""
    softmax_attention(q, k, v; mask=nothing) -> Array

Scaled dot-product attention.
q: (heads, head_dim, seq_q)
k: (heads, head_dim, seq_k)
v: (heads, head_dim, seq_k)
Returns: (heads, head_dim, seq_q)
"""
function softmax_attention(
    q::AbstractArray,
    k::AbstractArray,
    v::AbstractArray;
    mask::Union{AbstractArray,Nothing} = nothing,
)
    head_dim = size(q, 2)
    scale    = 1f0 / sqrt(Float32(head_dim))
    # Compute attention weights: (heads, seq_q, seq_k)
    attn_weights = batched_mul(permutedims(q, (1, 3, 2)), k) .* scale
    if mask !== nothing
        attn_weights = attn_weights .+ mask
    end
    attn_weights = softmax(attn_weights; dims=3)
    # Weighted sum: (heads, head_dim, seq_q)
    return permutedims(batched_mul(attn_weights, permutedims(v, (1, 3, 2))), (1, 3, 2))
end

"""
    gated_linear(x::AbstractArray, gate_weights::AbstractMatrix) -> AbstractArray

Gated linear unit: x * sigmoid(x * gate_weights).
"""
function gated_linear(x::AbstractArray, gate_weights::AbstractMatrix)
    gate = sigmoid.(x * gate_weights)
    return x .* gate
end

"""
    silu(x) -> same type

SiLU (Swish) activation: x * sigmoid(x).
"""
silu(x) = x .* sigmoid.(x)

# ──────────────────────────────────────────────
#  Triangle multiplication utilities
# ──────────────────────────────────────────────

"""
    triangle_multiply(
        a::AbstractArray,        # (L, L, c_z)
        left_proj::AbstractMatrix,   # (c_z, c_m)
        right_proj::AbstractMatrix,  # (c_z, c_m)
        gating::AbstractMatrix,      # (c_z, c_m)
        output_proj::AbstractMatrix, # (c_m, c_z)
        gating_linear::AbstractMatrix, # (c_z, c_z)
        center_norm_scale, center_norm_offset,
        left_norm_scale, left_norm_offset;
        outgoing::Bool = true
    )

One triangle multiplicative update step.
"""
function triangle_multiply(
    a::AbstractArray{T,3},
    left_proj::AbstractMatrix,
    right_proj::AbstractMatrix,
    gating::AbstractMatrix,
    output_proj::AbstractMatrix,
    gating_linear::AbstractMatrix,
    center_norm_scale::AbstractVector,
    center_norm_offset::AbstractVector,
    left_norm_scale::AbstractVector,
    left_norm_offset::AbstractVector;
    outgoing::Bool = true,
) where {T}
    L, _, c_z = size(a)
    c_m = size(left_proj, 2)

    # Layer norm input
    a_norm = apply_layer_norm(a, left_norm_scale, left_norm_offset)

    # Project to left and right
    # a_norm: (L, L, c_z)
    a_flat = reshape(a_norm, L*L, c_z)
    left   = reshape(a_flat * left_proj,  L, L, c_m)   # (L, L, c_m)
    right  = reshape(a_flat * right_proj, L, L, c_m)   # (L, L, c_m)
    gate   = sigmoid.(reshape(a_flat * gating, L, L, c_m))
    left   = left  .* gate
    right  = right .* gate

    # Triangle multiplication
    if outgoing
        # z_ij = Σ_k left_ik * right_jk
        # einsum ik,jk->ij over c_m
        result = zeros(T, L, L, c_m)
        for m in 1:c_m
            l_m = @view left[:, :, m]   # (L, L)
            r_m = @view right[:, :, m]  # (L, L)
            result[:, :, m] .= dropdims(sum(l_m .* permutedims(r_m, (2,1)); dims=3), dims=3)
        end
    else
        # Incoming: z_ij = Σ_k left_ki * right_kj
        result = zeros(T, L, L, c_m)
        for m in 1:c_m
            l_m = @view left[:, :, m]
            r_m = @view right[:, :, m]
            result[:, :, m] .= dropdims(sum(permutedims(l_m,(2,1)) .* r_m; dims=3), dims=3)
        end
    end

    # Center norm
    result_norm = apply_layer_norm(result, center_norm_scale, center_norm_offset)
    result_flat = reshape(result_norm, L*L, c_m)
    projected   = reshape(result_flat * output_proj, L, L, c_z)

    # Gating output
    gate_out = sigmoid.(reshape(a_flat * gating_linear, L, L, c_z))
    return projected .* gate_out
end

# ──────────────────────────────────────────────
#  Outer product mean
# ──────────────────────────────────────────────

"""
    outer_product_mean(
        msa::AbstractArray{T,3},   # (n_seqs, L, c_m)
        left_proj::AbstractMatrix, # (c_m, c_h)
        right_proj::AbstractMatrix,# (c_m, c_h)
        output_proj::AbstractMatrix# (c_h*c_h, c_z)
    ) -> Array{T,3}               # (L, L, c_z)

Compute outer product mean from MSA representation.
"""
function outer_product_mean(
    msa::AbstractArray{T,3},
    left_proj::AbstractMatrix,
    right_proj::AbstractMatrix,
    output_proj::AbstractMatrix,
) where {T}
    n_seqs, L, c_m = size(msa)
    c_h = size(left_proj, 2)

    msa_flat = reshape(msa, n_seqs * L, c_m)
    left     = reshape(msa_flat * left_proj,  n_seqs, L, c_h)  # (n, L, c_h)
    right    = reshape(msa_flat * right_proj, n_seqs, L, c_h)  # (n, L, c_h)

    # Outer product: for each (i,j), sum over n of left[:,i,:] ⊗ right[:,j,:]
    # Result: (L, L, c_h*c_h)
    op = zeros(T, L, L, c_h * c_h)
    for i in 1:L, j in 1:L
        l_ij = @view left[:, i, :]   # (n, c_h)
        r_ij = @view right[:, j, :]  # (n, c_h)
        op_ij = dropdims(mean(reshape(l_ij, n_seqs, c_h, 1) .* reshape(r_ij, n_seqs, 1, c_h); dims=1), dims=1)
        op[i, j, :] = vec(op_ij)
    end

    op_flat = reshape(op, L*L, c_h*c_h)
    out     = reshape(op_flat * output_proj, L, L, size(output_proj, 2))
    return out
end

# ──────────────────────────────────────────────
#  Relative position encoding
# ──────────────────────────────────────────────

"""
    compute_relative_position_encoding(
        token_index::Vector{Int32},
        token_chain_ids::Vector{String},
        max_relative_idx::Int = MAX_RELATIVE_IDX
    ) -> Matrix{Float32}   # (n_tokens, NUM_RELATIVE_POS_BINS)

Compute relative position one-hot encoding for pair representation.
"""
function compute_relative_position_encoding(
    token_index::Vector{Int32},
    token_chain_ids::Vector{String},
    max_relative_idx::Int = MAX_RELATIVE_IDX,
)::Matrix{Float32}
    n = length(token_index)
    n_bins = 2 * max_relative_idx + 2
    encoding = zeros(Float32, n * n, n_bins)

    for i in 1:n, j in 1:n
        idx = (i-1)*n + j
        if token_chain_ids[i] != token_chain_ids[j]
            # Different chain: all zeros (last bin used for same-chain indicator)
            encoding[idx, n_bins] = 0f0
        else
            rel_pos = clamp(Int(token_index[j]) - Int(token_index[i]), -max_relative_idx, max_relative_idx)
            bin_idx = rel_pos + max_relative_idx + 1
            encoding[idx, bin_idx] = 1f0
            encoding[idx, n_bins]  = 1f0  # same-chain indicator
        end
    end
    return reshape(encoding, n, n, n_bins)
end
