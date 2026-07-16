"""
Structure scoring functions: RMSD, superimposition, and related metrics.
"""

using LinearAlgebra
using Statistics

"""
    compute_rmsd(
        pred_positions::AbstractMatrix{Float32},
        true_positions::AbstractMatrix{Float32},
        mask::AbstractVector{Bool}
    ) -> Float32

Compute the (non-superimposed) RMSD between predicted and true positions
over the masked subset of atoms.
pred_positions, true_positions: (n_atoms, 3)
mask: length n_atoms, true for atoms to include
"""
function compute_rmsd(
    pred_positions::AbstractMatrix{Float32},
    true_positions::AbstractMatrix{Float32},
    mask::AbstractVector{Bool},
)::Float32
    n_total = size(pred_positions, 1)
    size(true_positions, 1) == n_total || error("Position array length mismatch")
    length(mask) == n_total || error("Mask length mismatch")

    n_masked = sum(mask)
    n_masked == 0 && return 0f0

    diff = (pred_positions[mask, :] .- true_positions[mask, :])
    return Float32(sqrt(sum(diff.^2) / n_masked))
end

"""
    compute_masked_rmsd(
        pred::AbstractMatrix{Float32},
        true::AbstractMatrix{Float32},
        mask::AbstractVector{Bool}
    ) -> Float32

Alias for compute_rmsd; computes RMSD only over masked positions.
"""
function compute_masked_rmsd(
    pred::AbstractMatrix{Float32},
    true::AbstractMatrix{Float32},
    mask::AbstractVector{Bool},
)::Float32
    return compute_rmsd(pred, true, mask)
end

"""
    superimpose(
        pred::AbstractMatrix{Float32},
        true_pos::AbstractMatrix{Float32},
        mask::AbstractVector{Bool}
    ) -> Tuple{Matrix{Float32}, Float32}

Superimpose pred onto true_pos over the masked atoms using the Kabsch algorithm.
Returns (aligned_pred, rmsd_after_alignment).

pred, true_pos: (n_atoms, 3)
mask: length n_atoms
"""
function superimpose(
    pred::AbstractMatrix{Float32},
    true_pos::AbstractMatrix{Float32},
    mask::AbstractVector{Bool},
)::Tuple{Matrix{Float32}, Float32}
    n = size(pred, 1)
    sum(mask) < 3 && return copy(pred), compute_rmsd(pred, true_pos, mask)

    pred_m = pred[mask, :]       # (m, 3)
    true_m = true_pos[mask, :]   # (m, 3)

    # Center both sets
    pred_center = mean(pred_m; dims=1)  # (1, 3)
    true_center = mean(true_m; dims=1)

    pred_centered = pred_m .- pred_center  # (m, 3)
    true_centered = true_m .- true_center

    # Covariance matrix
    H = pred_centered' * true_centered  # (3, 3)

    # SVD
    U, S, Vt = svd(H)
    V = Vt'

    # Correct reflection if needed (det < 0)
    d = det(V * U')
    D = diagm([1f0, 1f0, sign(d)])

    # Optimal rotation
    R = Float32.(V * D * U')

    # Apply rotation to all pred positions (centered)
    pred_all_centered = pred .- pred_center  # (n, 3)
    aligned = (R * pred_all_centered')' .+ true_center  # (n, 3)

    rmsd = compute_rmsd(Float32.(aligned), true_pos, mask)
    return Float32.(aligned), rmsd
end

"""
    compute_tm_score(
        pred_positions::AbstractMatrix{Float32},
        true_positions::AbstractMatrix{Float32},
        mask::AbstractVector{Bool};
        d0_override::Union{Float32,Nothing} = nothing
    ) -> Float32

Compute the TM-score between predicted and true positions (superimposed internally).
d0 is computed from the number of masked residues unless overridden.
"""
function compute_tm_score(
    pred_positions::AbstractMatrix{Float32},
    true_positions::AbstractMatrix{Float32},
    mask::AbstractVector{Bool};
    d0_override::Union{Float32,Nothing} = nothing,
)::Float32
    n_res = sum(mask)
    n_res < 1 && return 0f0

    d0 = if d0_override !== nothing
        d0_override
    else
        # TM-score d0 formula
        Float32(1.24 * (n_res - 15)^(1/3) - 1.8)
    end
    d0 = max(d0, 0.5f0)
    d0_sq = d0^2

    # Superimpose
    aligned, _ = superimpose(pred_positions, true_positions, mask)

    # Compute per-residue distances
    diff = aligned[mask, :] .- true_positions[mask, :]
    di_sq = vec(sum(diff.^2; dims=2))

    # TM-score formula
    tm = sum(1f0 ./ (1f0 .+ di_sq ./ d0_sq)) / Float32(n_res)
    return tm
end

"""
    compute_gdt_ts(
        pred_positions::AbstractMatrix{Float32},
        true_positions::AbstractMatrix{Float32},
        mask::AbstractVector{Bool}
    ) -> Float32

Compute GDT_TS score (average fraction of residues within 1, 2, 4, 8 Å
after optimal superposition).
"""
function compute_gdt_ts(
    pred_positions::AbstractMatrix{Float32},
    true_positions::AbstractMatrix{Float32},
    mask::AbstractVector{Bool},
)::Float32
    n_masked = sum(mask)
    n_masked < 1 && return 0f0

    aligned, _ = superimpose(pred_positions, true_positions, mask)
    diff = aligned[mask, :] .- true_positions[mask, :]
    di   = sqrt.(vec(sum(diff.^2; dims=2)))

    gdt = mean(Float32[mean(di .<= Float32(cutoff)) for cutoff in [1f0, 2f0, 4f0, 8f0]])
    return gdt
end
