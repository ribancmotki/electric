"""
Model-level MSA pairing interface.
Re-exports the data-level pairing functions for use in model featurisation.
This module provides the model-facing API for paired MSA construction.
"""

# The actual implementation lives in data/msa_pairing.jl; this module
# provides model-specific wrappers.

"""
    get_paired_msa_for_model(
        fold_input::FoldingInput,
        msa_config::MsaConfig
    ) -> Msa

Build the paired MSA for the full complex, ready for use in model featurisation.
Returns a concatenated Msa whose sequence column length equals the total token count.
"""
function get_paired_msa_for_model(
    fold_input::FoldingInput,
    msa_config::MsaConfig,
)::Msa
    per_chain_unpaired = Msa[]
    per_chain_paired   = Msa[]

    for chain in fold_input.protein_chains
        unpaired_a3m = chain.unpaired_msa
        paired_a3m   = chain.paired_msa

        unpaired = if unpaired_a3m !== nothing && !isempty(unpaired_a3m)
            Msa_from_a3m(unpaired_a3m)
        else
            Msa([uppercase(chain.sequence)], ["query"],
                zeros(Int, 1, length(chain.sequence)))
        end

        paired = if paired_a3m !== nothing && !isempty(paired_a3m)
            Msa_from_a3m(paired_a3m)
        else
            Msa(String[], String[], zeros(Int, 0, 0))
        end

        push!(per_chain_unpaired, truncate_msa(unpaired, msa_config.max_unpaired_sequences))
        push!(per_chain_paired,   truncate_msa(paired,   msa_config.max_paired_sequences))
    end

    for chain in fold_input.rna_chains
        unpaired_a3m = chain.unpaired_msa
        unpaired = if unpaired_a3m !== nothing && !isempty(unpaired_a3m)
            Msa_from_a3m(unpaired_a3m)
        else
            Msa([uppercase(chain.sequence)], ["query"],
                zeros(Int, 1, length(chain.sequence)))
        end
        push!(per_chain_unpaired, truncate_msa(unpaired, msa_config.max_unpaired_sequences))
        push!(per_chain_paired,   Msa(String[], String[], zeros(Int, 0, 0)))
    end

    if length(per_chain_unpaired) == 0
        return Msa(String[], String[], zeros(Int, 0, 0))
    elseif length(per_chain_unpaired) == 1
        return first(per_chain_unpaired)
    end

    # Multi-chain: build paired MSA
    paired_msa = build_paired_msa(per_chain_unpaired, per_chain_paired)

    # Concatenate unpaired MSAs column-wise for the unpaired block
    concat_unpaired = concat_msas_for_chains(per_chain_unpaired)

    # Merge paired and unpaired
    return merge_msas([paired_msa, concat_unpaired])
end
