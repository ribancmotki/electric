"""
Paired MSA construction for multi-chain (heteromeric) complexes.
"""

"""
    build_paired_msa(
        per_chain_unpaired_msas::Vector{Msa},
        per_chain_paired_msas::Vector{Msa}
    ) -> Msa

Construct the full paired MSA matrix for a multi-chain complex.

For each unique pairing key (UniProt/UniRef ID) that appears in at least two chains,
insert a paired row with sequences from each matching chain.
Chains where no match is found get gap sequences.

Returns a single Msa whose sequences are concatenations of per-chain sequences,
with pairing by shared sequence identifiers.
"""
function build_paired_msa(
    per_chain_unpaired_msas::Vector{Msa},
    per_chain_paired_msas::Vector{Msa}
)::Msa
    n_chains = length(per_chain_unpaired_msas)
    n_chains == 0 && return Msa(String[], String[], zeros(Int, 0, 0))

    # Get per-chain alignment lengths for gap padding
    per_chain_len = Int[alignment_length(msa) for msa in per_chain_unpaired_msas]

    # Build pairing key → (chain_idx, seq_idx) maps from paired MSAs
    pairing_maps = Vector{Dict{String,Int}}(undef, n_chains)
    for c in 1:n_chains
        pairing_maps[c] = Dict{String,Int}()
        if c <= length(per_chain_paired_msas)
            msa = per_chain_paired_msas[c]
            for i in 1:n_seqs(msa)
                ids = get_identifiers(msa.descriptions[i])
                key = get_pairing_key(ids)
                key !== nothing && !haskey(pairing_maps[c], key) && (pairing_maps[c][key] = i)
            end
        end
    end

    # Find all pairing keys that appear in at least 2 chains
    all_keys = Set{String}()
    for c in 1:n_chains
        union!(all_keys, keys(pairing_maps[c]))
    end

    paired_seqs  = String[]
    paired_descs = String[]
    paired_dels  = Vector{Int}[]

    for key in sort(collect(all_keys))
        # Check how many chains have this key
        chain_presence = [haskey(pairing_maps[c], key) for c in 1:n_chains]
        sum(chain_presence) < 2 && continue

        # Build concatenated row
        row_parts  = String[]
        desc_parts = String[]
        del_parts  = Int[]

        for c in 1:n_chains
            seq_len = per_chain_len[c]
            if chain_presence[c] && c <= length(per_chain_paired_msas)
                seq_idx = pairing_maps[c][key]
                msa     = per_chain_paired_msas[c]
                push!(row_parts, msa.sequences[seq_idx])
                push!(desc_parts, msa.descriptions[seq_idx])
                if size(msa.deletion_matrix, 1) >= seq_idx
                    append!(del_parts, msa.deletion_matrix[seq_idx, :])
                else
                    append!(del_parts, zeros(Int, seq_len))
                end
            else
                # Gap row for chains without a match
                push!(row_parts, '-'^seq_len)
                push!(desc_parts, "gap_$key")
                append!(del_parts, zeros(Int, seq_len))
            end
        end

        push!(paired_seqs,  join(row_parts))
        push!(paired_descs, join(desc_parts, ";"))
        push!(paired_dels,  del_parts)
    end

    isempty(paired_seqs) && return Msa(String[], String[], zeros(Int, 0, 0))

    total_len = sum(per_chain_len)
    n_paired  = length(paired_seqs)
    del_mat   = zeros(Int, n_paired, total_len)
    for (i, row) in enumerate(paired_dels)
        len_r = length(row)
        if len_r >= total_len
            del_mat[i, :] = row[1:total_len]
        else
            del_mat[i, 1:len_r] = row
        end
    end

    return Msa(paired_seqs, paired_descs, del_mat)
end

"""
    concat_msas_for_chains(msas::Vector{Msa}) -> Msa

Concatenate per-chain MSAs column-wise (for building the unpaired MSA block
of the concatenated input).
All MSAs must have the same number of rows.
Each chain's query is at row 1; other rows are padded with gaps for other chains.
"""
function concat_msas_for_chains(msas::Vector{Msa})::Msa
    n_chains = length(msas)
    n_chains == 0 && return Msa(String[], String[], zeros(Int, 0, 0))
    n_chains == 1 && return first(msas)

    per_chain_len = Int[alignment_length(msa) for msa in msas]
    total_len = sum(per_chain_len)

    all_seqs  = String[]
    all_descs = String[]
    all_dels  = Vector{Int}[]

    # First row: concatenated query sequences
    query_parts = String[]
    query_del   = Int[]
    for (c, msa) in enumerate(msas)
        if n_seqs(msa) >= 1
            push!(query_parts, msa.sequences[1])
            if size(msa.deletion_matrix, 1) >= 1
                append!(query_del, msa.deletion_matrix[1, :])
            else
                append!(query_del, zeros(Int, per_chain_len[c]))
            end
        else
            push!(query_parts, '-'^per_chain_len[c])
            append!(query_del, zeros(Int, per_chain_len[c]))
        end
    end
    push!(all_seqs,  join(query_parts))
    push!(all_descs, "query")
    push!(all_dels,  query_del)

    # For each chain, add its non-query rows padded with gaps elsewhere
    for c in 1:n_chains
        msa = msas[c]
        for i in 2:n_seqs(msa)
            row_parts = String[]
            del_parts = Int[]
            for c2 in 1:n_chains
                if c2 == c
                    push!(row_parts, msa.sequences[i])
                    if size(msa.deletion_matrix, 1) >= i
                        append!(del_parts, msa.deletion_matrix[i, :])
                    else
                        append!(del_parts, zeros(Int, per_chain_len[c2]))
                    end
                else
                    push!(row_parts, '-'^per_chain_len[c2])
                    append!(del_parts, zeros(Int, per_chain_len[c2]))
                end
            end
            push!(all_seqs,  join(row_parts))
            push!(all_descs, msa.descriptions[i])
            push!(all_dels,  del_parts)
        end
    end

    n = length(all_seqs)
    del_mat = zeros(Int, n, total_len)
    for (i, row) in enumerate(all_dels)
        len_r = length(row)
        if len_r >= total_len
            del_mat[i, :] = row[1:total_len]
        else
            del_mat[i, 1:len_r] = row
        end
    end

    return Msa(all_seqs, all_descs, del_mat)
end
