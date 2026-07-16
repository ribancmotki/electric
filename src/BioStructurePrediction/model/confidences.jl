"""
Confidence metric computation from model output logits.
"""

using Statistics
using LinearAlgebra

# ──────────────────────────────────────────────
#  pLDDT
# ──────────────────────────────────────────────

"""
    compute_plddt(plddt_logits::AbstractArray{Float32,3}) -> Vector{Float32}

Compute per-atom pLDDT scores (0–100) from logits.

plddt_logits: (num_tokens, num_atom_types, num_bins) = (T, 24, 50)
Returns: (T * 24,) vector of pLDDT values (0–100)
"""
function compute_plddt(plddt_logits::AbstractArray{Float32,3})::Vector{Float32}
    n_tokens, n_atom_types, n_bins = size(plddt_logits)
    # Softmax over bins
    probs = softmax_last_dim(plddt_logits)  # (T, 24, 50)
    # Expected value over bin centers
    bin_centers = PLDDT_BIN_CENTERS  # (50,)
    # plddt: sum over bins of prob * bin_center, scaled to 0-100
    # (T, 24, 50) * (50,) -> (T, 24)
    plddt_raw = reshape(
        reshape(probs, n_tokens * n_atom_types, n_bins) * bin_centers,
        n_tokens, n_atom_types,
    )
    # Scale to 0-100
    plddt_100 = plddt_raw .* 100f0
    # Return flattened
    return vec(plddt_100)
end

"""
    compute_per_token_plddt(plddt_logits::AbstractArray{Float32,3}) -> Vector{Float32}

Compute mean pLDDT per token (averaged over present atom types).
Returns: (num_tokens,) vector.
"""
function compute_per_token_plddt(plddt_logits::AbstractArray{Float32,3})::Vector{Float32}
    n_tokens, n_atom_types, _ = size(plddt_logits)
    all_plddt = reshape(compute_plddt(plddt_logits), n_tokens, n_atom_types)
    return vec(mean(all_plddt; dims=2))
end

# ──────────────────────────────────────────────
#  PAE
# ──────────────────────────────────────────────

"""
    compute_pae(pae_logits::AbstractArray{Float32,3}) -> Matrix{Float32}

Compute PAE (Predicted Aligned Error) matrix in Ångströms.

pae_logits: (num_tokens, num_tokens, num_bins) = (T, T, 64)
Returns: (T, T) matrix in Å
"""
function compute_pae(pae_logits::AbstractArray{Float32,3})::Matrix{Float32}
    n_tokens, _, n_bins = size(pae_logits)
    probs = softmax_last_dim(pae_logits)  # (T, T, 64)
    bin_centers = PAE_BIN_CENTERS  # (64,)
    pae_flat = reshape(probs, n_tokens * n_tokens, n_bins) * bin_centers
    return reshape(pae_flat, n_tokens, n_tokens)
end

# ──────────────────────────────────────────────
#  pTM / ipTM
# ──────────────────────────────────────────────

"""
    compute_ptm(
        pae_logits::AbstractArray{Float32,3},
        token_chain_ids::Vector{String};
        interface_only::Bool = false
    ) -> Float64

Compute pTM or ipTM from PAE logits.

If interface_only=true, compute ipTM (restrict to inter-chain token pairs).
"""
function compute_ptm(
    pae_logits::AbstractArray{Float32,3},
    token_chain_ids::Vector{String};
    interface_only::Bool = false,
)::Float64
    n = size(pae_logits, 1)
    n < 1 && return 0.0

    pae = compute_pae(pae_logits)  # (n, n)

    # d0 depends on sequence length
    d0 = compute_d0(n)
    d0_sq = Float64(d0)^2

    # Mask for which pairs to include
    if interface_only
        # ipTM: only inter-chain pairs
        mask = [token_chain_ids[i] != token_chain_ids[j] for i in 1:n, j in 1:n]
        n_pairs = sum(mask)
        n_pairs < 1 && return 0.0
    else
        # pTM: all pairs
        mask = trues(n, n)
        n_pairs = n * n
    end

    # TM-score formula using PAE as the distance estimate
    score_matrix = 1.0 ./ (1.0 .+ (Float64.(pae).^2) ./ d0_sq)
    score_matrix_masked = score_matrix .* mask

    # pTM = max over aligned residue j of (1/n * sum_i TM_score(i, j | aligned to j))
    # Per the original pTM formula: max_j (1/n * sum_i 1/(1 + d_ij^2/d0^2))
    col_sums = vec(sum(score_matrix_masked; dims=1))  # (n,)
    ptm = maximum(col_sums) / Float64(n)
    return min(ptm, 1.0)
end

"""
    compute_d0(n_res::Int) -> Float32

Compute the d0 parameter for the TM-score formula.
"""
function compute_d0(n_res::Int)::Float32
    n_res <= 21 && return 0.5f0
    return Float32(1.24 * (n_res - 15)^(1/3) - 1.8)
end

"""
    compute_per_chain_ptm(
        pae_logits::AbstractArray{Float32,3},
        token_chain_ids::Vector{String}
    ) -> Dict{String,Float64}

Compute pTM for each chain separately.
"""
function compute_per_chain_ptm(
    pae_logits::AbstractArray{Float32,3},
    token_chain_ids::Vector{String},
)::Dict{String,Float64}
    chain_ids = unique(token_chain_ids)
    result = Dict{String,Float64}()
    for cid in chain_ids
        chain_mask = findall(==(cid), token_chain_ids)
        length(chain_mask) < 1 && continue
        sub_logits = pae_logits[chain_mask, chain_mask, :]
        result[cid] = compute_ptm(sub_logits, token_chain_ids[chain_mask]; interface_only=false)
    end
    return result
end

"""
    compute_per_chain_pair_iptm(
        pae_logits::AbstractArray{Float32,3},
        token_chain_ids::Vector{String}
    ) -> Dict{String,Float64}

Compute ipTM for each ordered pair of chains.
"""
function compute_per_chain_pair_iptm(
    pae_logits::AbstractArray{Float32,3},
    token_chain_ids::Vector{String},
)::Dict{String,Float64}
    chain_ids = unique(token_chain_ids)
    result = Dict{String,Float64}()
    for c1 in chain_ids, c2 in chain_ids
        c1 == c2 && continue
        key = "$c1,$c2"
        mask1 = findall(==(c1), token_chain_ids)
        mask2 = findall(==(c2), token_chain_ids)
        (isempty(mask1) || isempty(mask2)) && continue
        # Sub-logits for cross-chain pairs (rows=c1, cols=c2)
        all_mask = vcat(mask1, mask2)
        sub_logits = pae_logits[all_mask, all_mask, :]
        sub_chain_ids = vcat(fill(c1, length(mask1)), fill(c2, length(mask2)))
        result[key] = compute_ptm(sub_logits, sub_chain_ids; interface_only=true)
    end
    return result
end

# ──────────────────────────────────────────────
#  Clash detection
# ──────────────────────────────────────────────

"""
    detect_clashes(
        s::Structure;
        tolerance::Float32 = 0.4f0,
    ) -> Bool

Return true if any pair of atoms from different residues is closer than
their VDW radii sum minus tolerance.
"""
function detect_clashes(
    s::Structure;
    tolerance::Float32 = 0.4f0,
)::Bool
    n = num_atoms(s)
    n <= 1 && return false

    xs = get_column(s.atoms, :Cartn_x)
    ys = get_column(s.atoms, :Cartn_y)
    zs = get_column(s.atoms, :Cartn_z)
    elems     = get_column(s.atoms, :type_symbol)
    chain_ids = get_column(s.atoms, :label_asym_id)
    seq_ids   = get_column(s.atoms, :label_seq_id)

    # Check a subsample for efficiency (full O(n^2) is slow for large structures)
    step = max(1, n ÷ 500)
    for i in 1:step:n
        for j in i+1:step:n
            # Skip atoms in the same residue
            if chain_ids[i] == chain_ids[j] && seq_ids[i] == seq_ids[j]
                continue
            end
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dz = zs[i] - zs[j]
            d  = sqrt(dx^2 + dy^2 + dz^2)
            min_d = get_vdw_radius(elems[i]) + get_vdw_radius(elems[j]) - tolerance
            d < min_d && return true
        end
    end
    return false
end

# ──────────────────────────────────────────────
#  Ranking score
# ──────────────────────────────────────────────

"""
    compute_ranking_score(
        iptm::Float64,
        ptm::Float64,
        plddt_per_token::Vector{Float32},
        has_clash::Bool,
    ) -> Float64

Compute the ranking score:
  0.8 × ipTM + 0.2 × pTM + 0.5 × disorder_score − 100 × has_clash

where disorder_score = fraction of tokens with pLDDT < 50.
"""
function compute_ranking_score(
    iptm::Float64,
    ptm::Float64,
    plddt_per_token::Vector{Float32},
    has_clash::Bool,
)::Float64
    disorder_score = isempty(plddt_per_token) ? 0.0 :
        Float64(mean(plddt_per_token .< 50f0))
    clash_penalty = has_clash ? 100.0 : 0.0
    return 0.8 * iptm + 0.2 * ptm + 0.5 * disorder_score - clash_penalty
end

# ──────────────────────────────────────────────
#  Experimentally resolved
# ──────────────────────────────────────────────

"""
    compute_experimentally_resolved(
        er_logits::AbstractArray{Float32,3}
    ) -> Vector{Float32}

Compute per-atom experimentally resolved probability from logits.
er_logits: (num_tokens, 24, 2)
Returns: (num_tokens * 24,) probabilities (0-1) that atom is resolved.
"""
function compute_experimentally_resolved(
    er_logits::AbstractArray{Float32,3}
)::Vector{Float32}
    n_tokens, n_atom_types, _ = size(er_logits)
    probs = softmax_last_dim(er_logits)  # (T, 24, 2)
    # Channel 2 = resolved probability
    return vec(probs[:, :, 2])
end

# ──────────────────────────────────────────────
#  Full confidence computation
# ──────────────────────────────────────────────

"""
    compute_confidence_metrics(
        result::Dict{String,Any},
        predicted_structure::Structure,
        token_chain_ids::Vector{String},
    ) -> ConfidenceMetrics

Compute all confidence metrics from model result dict.
"""
function compute_confidence_metrics(
    result::Dict{String,Any},
    predicted_structure::Structure,
    token_chain_ids::Vector{String},
)::ConfidenceMetrics
    n_tokens = length(token_chain_ids)

    # pLDDT
    plddt_logits = get(result, "plddt_logits", nothing)
    if plddt_logits !== nothing && ndims(plddt_logits) == 3
        all_plddt = compute_plddt(Float32.(plddt_logits))
        per_token_plddt = compute_per_token_plddt(Float32.(plddt_logits))
    else
        all_plddt       = fill(50f0, n_tokens * NUM_ATOM_TYPES_PLDDT)
        per_token_plddt = fill(50f0, n_tokens)
    end

    # PAE
    pae_logits = get(result, "pae_logits", nothing)
    pae_matrix = if pae_logits !== nothing && ndims(pae_logits) == 3
        compute_pae(Float32.(pae_logits))
    else
        zeros(Float32, n_tokens, n_tokens)
    end

    # pTM / ipTM
    if pae_logits !== nothing && ndims(pae_logits) == 3
        ptm  = compute_ptm(Float32.(pae_logits), token_chain_ids; interface_only=false)
        iptm = compute_ptm(Float32.(pae_logits), token_chain_ids; interface_only=true)
        chain_ptm       = compute_per_chain_ptm(Float32.(pae_logits), token_chain_ids)
        chain_pair_iptm = compute_per_chain_pair_iptm(Float32.(pae_logits), token_chain_ids)
    else
        ptm  = 0.0
        iptm = 0.0
        chain_ptm       = Dict{String,Float64}()
        chain_pair_iptm = Dict{String,Float64}()
    end

    # Clash
    has_clash = detect_clashes(predicted_structure)

    # Ranking score
    rs = compute_ranking_score(iptm, ptm, per_token_plddt, has_clash)

    # Disorder
    disorder = isempty(per_token_plddt) ? 0.0 :
        Float64(mean(per_token_plddt .< 50f0))

    # Experimentally resolved
    er_logits = get(result, "experimentally_resolved_logits", nothing)
    er_probs  = if er_logits !== nothing && ndims(er_logits) == 3
        compute_experimentally_resolved(Float32.(er_logits))
    else
        fill(0f0, n_tokens * NUM_ATOM_TYPES_PLDDT)
    end

    return ConfidenceMetrics(
        all_plddt,
        per_token_plddt,
        pae_matrix,
        ptm,
        iptm,
        rs,
        chain_ptm,
        chain_pair_iptm,
        er_probs,
        disorder,
        has_clash,
    )
end

# ──────────────────────────────────────────────
#  Utility: softmax over last dimension
# ──────────────────────────────────────────────

"""
    softmax_last_dim(x::AbstractArray) -> same shape

Apply softmax over the last dimension.
"""
function softmax_last_dim(x::AbstractArray{T}) where {T<:AbstractFloat}
    max_x = maximum(x; dims=ndims(x))
    e     = exp.(x .- max_x)
    return e ./ sum(e; dims=ndims(x))
end
