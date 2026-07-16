"""
Full featurisation pipeline: converts FoldingInput into model-ready BatchDicts.
"""

using Dates

"""
    featurise_input(
        fold_input::FoldingInput;
        buckets::Union{Vector{Int},Nothing},
        ccd::Ccd,
        verbose::Bool = false,
        ref_max_modified_date::Union{Date,Nothing} = nothing,
        conformer_max_iterations::Union{Int,Nothing} = nothing,
        resolve_msa_overlaps::Bool = true
    ) -> Vector{BatchDict}

Featurise a FoldingInput into one BatchDict per seed.

Token count:
- Protein/RNA/DNA: 1 token per residue
- Ligand: 1 token per heavy atom (from CCD or SMILES)
"""
function featurise_input(
    fold_input::FoldingInput;
    buckets::Union{Vector{Int},Nothing}          = nothing,
    ccd::Ccd,
    verbose::Bool                                 = false,
    ref_max_modified_date::Union{Date,Nothing}    = nothing,
    conformer_max_iterations::Union{Int,Nothing}  = nothing,
    resolve_msa_overlaps::Bool                    = true,
)::Vector{BatchDict}
    msa_config = default_msa_config()

    # ── Step 1: Compute per-chain features ─────────────────────────────────────
    per_chain_data = Dict{String,Any}[]

    for chain in fold_input.protein_chains
        for cid in chain.ids
            d = process_protein_chain_features(chain, ccd, msa_config)
            d["chain_ids"] = [cid]
            d["entity_type"] = :protein
            push!(per_chain_data, d)
        end
    end

    for chain in fold_input.rna_chains
        for cid in chain.ids
            d = process_rna_chain_features(chain, ccd, msa_config)
            d["chain_ids"] = [cid]
            d["entity_type"] = :rna
            push!(per_chain_data, d)
        end
    end

    for chain in fold_input.dna_chains
        for cid in chain.ids
            d = process_dna_chain_features(chain, ccd)
            d["chain_ids"] = [cid]
            d["entity_type"] = :dna
            push!(per_chain_data, d)
        end
    end

    for lig in fold_input.ligands
        for cid in lig.ids
            d = process_ligand_features(lig, ccd;
                conformer_max_iterations = conformer_max_iterations,
                ref_max_modified_date    = ref_max_modified_date,
            )
            d["chain_ids"] = [cid]
            d["entity_type"] = :ligand
            push!(per_chain_data, d)
        end
    end

    if isempty(per_chain_data)
        error("FoldingInput has no chains: $(fold_input.name)")
    end

    # ── Step 2: Count tokens ───────────────────────────────────────────────────
    n_tokens = sum(length(d["residue_types"]) for d in per_chain_data)
    verbose && @info "Total token count: $n_tokens"

    # ── Step 3: Select bucket ──────────────────────────────────────────────────
    bucket_size = if buckets === nothing || isempty(buckets)
        n_tokens
    else
        select_bucket(n_tokens, buckets)
    end
    verbose && @info "Using bucket size: $bucket_size"

    # ── Step 4: Build one BatchDict per seed ───────────────────────────────────
    batch_dicts = BatchDict[]
    for seed in fold_input.rng_seeds
        batch = assemble_batch(per_chain_data, fold_input, seed, bucket_size)
        push!(batch_dicts, batch)
    end

    return batch_dicts
end

"""
    compute_num_tokens(fold_input::FoldingInput, ccd::Ccd) -> Int

Compute the number of tokens for a FoldingInput without full featurisation.
"""
function compute_num_tokens(fold_input::FoldingInput, ccd::Ccd)::Int
    n = 0
    for chain in fold_input.protein_chains
        n += length(chain.sequence) * length(chain.ids)
    end
    for chain in fold_input.rna_chains
        n += length(chain.sequence) * length(chain.ids)
    end
    for chain in fold_input.dna_chains
        n += length(chain.sequence) * length(chain.ids)
    end
    for lig in fold_input.ligands
        for code in lig.ccd_codes
            comp = get_component(ccd, code)
            if comp !== nothing
                n_heavy = count(a -> !a.leaving_atom && a.element != "H", comp.atoms)
                n += n_heavy * length(lig.ids)
            else
                n += length(lig.ids)  # fallback: 1 token per ligand ID
            end
        end
        if isempty(lig.ccd_codes) && lig.smiles !== nothing
            _, _, n_atoms = parse_smiles_atoms(lig.smiles)
            n += n_atoms * length(lig.ids)
        end
    end
    return n
end
