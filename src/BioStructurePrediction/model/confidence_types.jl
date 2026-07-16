"""
Type definitions for confidence metrics.
"""

"""
    ConfidenceMetrics

Complete confidence metrics for a single predicted structure.
"""
struct ConfidenceMetrics
    # Per-atom pLDDT scores (0–100 scale)
    plddt::Vector{Float32}
    # Per-token mean pLDDT (averaged over present atoms)
    mean_plddt_per_token::Vector{Float32}
    # PAE matrix: (num_tokens, num_tokens), in Ångströms
    pae::Matrix{Float32}
    # Global pTM score (0–1)
    ptm::Float64
    # Interface pTM (0–1)
    iptm::Float64
    # Ranking score
    ranking_score::Float64
    # Per-chain pTM
    chain_ptm::Dict{String,Float64}
    # Per-chain-pair ipTM
    chain_pair_iptm::Dict{String,Float64}
    # Per-atom experimentally resolved probability (0–1)
    experimentally_resolved::Vector{Float32}
    # Disorder score: fraction of tokens with pLDDT < 50
    fraction_disordered::Float64
    # Whether any inter-residue atom pair is closer than VDW sum - tolerance
    has_clash::Bool
end

"""
    SummaryConfidences

Summary confidence metrics for quick overview.
"""
struct SummaryConfidences
    ptm::Float64
    iptm::Float64
    ranking_score::Float64
    mean_plddt::Float64
    fraction_disordered::Float64
    has_clash::Bool
end

"""
    InferenceResult

Complete output from a single diffusion sample.
"""
struct InferenceResult
    predicted_structure::Structure
    confidence_metrics::ConfidenceMetrics
    metadata::Dict{String,Any}
end

"""
    ResultsForSeed

All results for a single RNG seed.
"""
struct ResultsForSeed
    seed::Int
    inference_results::Vector{InferenceResult}
    full_fold_input::FoldingInput
    embeddings::Union{Dict{String,Array},Nothing}
    distogram::Union{Array,Nothing}
end

"""
    ranking_score(ir::InferenceResult) -> Float64

Return the ranking score for an InferenceResult.
"""
function ranking_score(ir::InferenceResult)::Float64
    return ir.confidence_metrics.ranking_score
end
