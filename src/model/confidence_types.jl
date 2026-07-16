"""
confidence_types.jl — Result and confidence data types.
"""

using Dates

# ──────────────────────────────────────────────────────────────────────────────
# Confidence metric types
# ──────────────────────────────────────────────────────────────────────────────

"""
    ConfidenceMetrics

Per-structure confidence metrics from the model's confidence head.
"""
struct ConfidenceMetrics
    predicted_lddt::Array{Float32}          # (num_tokens, max_atoms_per_token)
    predicted_pae::Array{Float32}            # (num_tokens, num_tokens)
    predicted_pde::Array{Float32}            # (num_tokens, num_tokens)
    predicted_tm_score::Float32
    predicted_iptm::Float32
    ranking_score::Float32
    fraction_disordered::Float32
    has_clash::Bool
    chain_pair_pae_min::Union{Array{Float32},Nothing}
    chain_pair_pde_mean::Union{Array{Float32},Nothing}
    experimentally_resolved::Array{Float32}  # (num_tokens, max_atoms_per_token)
end

function Base.show(io::IO, cm::ConfidenceMetrics)
    print(io, "ConfidenceMetrics(pTM=$(cm.predicted_tm_score), ipTM=$(cm.predicted_iptm), ranking=$(cm.ranking_score))")
end

"""
    SummaryConfidences

Summary-level confidence scores for a predicted structure.
"""
struct SummaryConfidences
    ptm::Float32
    iptm::Float32
    ranking_score::Float32
    chain_pair_iptm::Union{Array{Float32},Nothing}
    chain_iptm::Union{Vector{Float32},Nothing}
    has_inter_chain_predicted_contacts::Bool
    mean_plddt::Float32
end

function Base.show(io::IO, sc::SummaryConfidences)
    print(io, "SummaryConfidences(pTM=$(sc.ptm), ranking=$(sc.ranking_score), mean_pLDDT=$(sc.mean_plddt))")
end

# ──────────────────────────────────────────────────────────────────────────────
# Inference result types
# ──────────────────────────────────────────────────────────────────────────────

"""
    InferenceResult

Full output of a single inference run (one seed × one sample).
"""
struct InferenceResult
    predicted_structure::Structure
    numerical_data::Dict{String,Any}
    metadata::Dict{String,Any}
    debug_outputs::Dict{String,Any}
    model_id::Vector{UInt8}
end

function Base.show(io::IO, ir::InferenceResult)
    rs = get(ir.numerical_data, "ranking_score", NaN)
    print(io, "InferenceResult(ranking_score=$(round(rs, digits=4)))")
end

"""
    ResultsForSeed

All inference results for a single RNG seed.
"""
struct ResultsForSeed
    seed::Int
    inference_results::Vector{InferenceResult}
    embeddings::Union{Dict{String,Array},Nothing}
end

function Base.show(io::IO, r::ResultsForSeed)
    print(io, "ResultsForSeed(seed=$(r.seed), $(length(r.inference_results)) samples)")
end

# ──────────────────────────────────────────────────────────────────────────────
# Model result type alias
# ──────────────────────────────────────────────────────────────────────────────

"""
    ModelResult

Dict representing the full raw output of the model forward pass.
"""
const ModelResult = Dict{String,Any}

"""
    make_empty_model_result() -> ModelResult

Create an empty model result dict.
"""
function make_empty_model_result()::ModelResult
    return ModelResult()
end
