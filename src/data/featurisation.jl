"""
featurisation.jl — Entry point for featurizing folding inputs into model batches.
"""

using Logging

# Type alias for model input batch
const BatchDict = Dict{String,AbstractArray}

"""
    featurise_input(fold_input::Input, ccd::Ccd,
                    buckets::Union{Vector{Int},Nothing}=nothing;
                    ref_max_modified_date=nothing,
                    conformer_max_iterations=nothing,
                    resolve_msa_overlaps=true,
                    verbose=false) -> Vector{BatchDict}

Featurize a folding input into model-ready BatchDicts, one per rng_seed.
"""
function featurise_input(
    fold_input::Input,
    ccd::Ccd,
    buckets::Union{Vector{Int},Nothing} = nothing;
    ref_max_modified_date = nothing,
    conformer_max_iterations::Union{Int,Nothing} = nothing,
    resolve_msa_overlaps::Bool = true,
    verbose::Bool = false,
)::Vector{BatchDict}
    # Validate inputs
    for chain in fold_input.chains
        if chain isa ProteinChain
            chain.unpaired_msa === nothing &&
                @warn "ProteinChain $(chain.id) has no unpaired_msa; featurisation may be incomplete"
        elseif chain isa RnaChain
            chain.unpaired_msa === nothing &&
                @warn "RnaChain $(chain.id) has no unpaired_msa"
        end
    end

    pipeline_cfg = WholePdbPipelineConfig(
        buckets = buckets,
        ref_max_modified_date = ref_max_modified_date,
        conformer_max_iterations = conformer_max_iterations,
        resolve_msa_overlaps = resolve_msa_overlaps,
    )

    results = BatchDict[]
    for seed in fold_input.rng_seeds
        rng = Random.MersenneTwister(seed)
        try
            batch = process_item(pipeline_cfg, fold_input, ccd, rng, seed)
            push!(results, batch)
        catch e
            @error "Featurisation failed for seed $seed: $e"
            rethrow(e)
        end
    end
    return results
end
