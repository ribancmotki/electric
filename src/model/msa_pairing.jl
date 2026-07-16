"""
msa_pairing.jl — Inter-chain MSA pairing for multimer modeling.
"""

using Logging
using Random

# ──────────────────────────────────────────────────────────────────────────────
# Pairing strategy
# ──────────────────────────────────────────────────────────────────────────────

"""
    pair_msas(chain_msas::Dict{String,Msa};
              max_paired_seqs::Int=1152,
              pairing_db::String="uniprot") -> Dict{String,Msa}

Pair MSAs from multiple chains by species ID (TaxID-based pairing).
Returns a dict of chain_id → paired Msa (same-length, re-indexed).
"""
function pair_msas(
    chain_msas::Dict{String,Msa};
    max_paired_seqs::Int = 1152,
    pairing_db::String   = "uniprot",
)::Dict{String,Msa}
    chain_ids = sort(collect(keys(chain_msas)))
    length(chain_ids) <= 1 && return chain_msas  # no pairing needed

    # Build species → sequence index map per chain
    chain_species_maps = Dict{String,Dict{String,Int}}()
    for cid in chain_ids
        msa = chain_msas[cid]
        n = n_seqs(msa)
        species_map = Dict{String,Int}()
        for i in 2:n  # skip query at index 1
            identifiers = get_identifiers(msa.descriptions[i])
            key = get_pairing_key(identifiers)
            isempty(key) && continue
            haskey(species_map, key) || (species_map[key] = i)
        end
        chain_species_maps[cid] = species_map
    end

    # Find species present in ALL chains
    species_in_all = intersect([Set(keys(m)) for m in values(chain_species_maps)]...)

    @debug "MSA pairing: $(length(species_in_all)) species common to all $(length(chain_ids)) chains"

    if isempty(species_in_all)
        @info "No common species found for MSA pairing; returning unpaired MSAs"
        return chain_msas
    end

    # Build paired MSAs: query first, then paired rows in consistent order
    species_order = sort(collect(species_in_all))
    n_pairs = min(max_paired_seqs - 1, length(species_order))
    species_order = species_order[1:n_pairs]

    result = Dict{String,Msa}()
    for cid in chain_ids
        msa = chain_msas[cid]
        sp_map = chain_species_maps[cid]
        L = alignment_length(msa)

        # Always include query at position 1
        new_seqs  = [msa.sequences[1]]
        new_descs = [msa.descriptions[1]]
        new_del   = size(msa.deletion_matrix, 1) > 0 ?
                    [msa.deletion_matrix[1:1, :]] : [zeros(Int, 1, L)]

        for sp in species_order
            idx = get(sp_map, sp, 0)
            if idx > 0 && idx <= n_seqs(msa)
                push!(new_seqs,  msa.sequences[idx])
                push!(new_descs, msa.descriptions[idx])
                push!(new_del, msa.deletion_matrix[idx:idx, :])
            else
                # Gap row (all dashes)
                push!(new_seqs,  repeat("-", L))
                push!(new_descs, "gap_$sp")
                push!(new_del, zeros(Int, 1, L))
            end
        end

        del_mat = vcat(new_del...)

        result[cid] = Msa(
            new_seqs, new_descs, del_mat,
            msa.query_sequence, msa.chain_poly_type,
        )
    end

    return result
end

"""
    deduplicate_paired_msas(paired_msas::Dict{String,Msa}) -> Dict{String,Msa}

Remove rows where all chains have the same species-padded gap sequence.
"""
function deduplicate_paired_msas(paired_msas::Dict{String,Msa})::Dict{String,Msa}
    isempty(paired_msas) && return paired_msas
    chain_ids = collect(keys(paired_msas))

    # All MSAs must have the same number of sequences after pairing
    n_seqs_per_chain = [n_seqs(paired_msas[cid]) for cid in chain_ids]
    allequal(n_seqs_per_chain) || return paired_msas

    n = n_seqs_per_chain[1]
    keep = trues(n)

    for i in 2:n
        # Row is all-gaps across all chains?
        all_gap = all(all(c == '-' for c in paired_msas[cid].sequences[i]) for cid in chain_ids)
        all_gap && (keep[i] = false)
    end

    result = Dict{String,Msa}()
    for cid in chain_ids
        m = paired_msas[cid]
        kidxs = findall(keep)
        L = alignment_length(m)
        del_mat_rows = [m.deletion_matrix[i:i, :] for i in kidxs]
        result[cid] = Msa(
            m.sequences[kidxs],
            m.descriptions[kidxs],
            isempty(del_mat_rows) ? zeros(Int, 0, L) : vcat(del_mat_rows...),
            m.query_sequence,
            m.chain_poly_type,
        )
    end
    return result
end

function allequal(v)
    length(v) <= 1 && return true
    return all(==(v[1]), v)
end
