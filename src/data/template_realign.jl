"""
template_realign.jl — Template hit realignment using hmmalign and Smith-Waterman.
"""

using Logging

"""
    realign_template(query_sequence::String, template_sequence::String,
                     hit_alignment::String, hmm_profile::String;
                     hmmalign_binary::String="hmmalign",
                     hmmbuild_binary::String="hmmbuild") -> Dict{Int,Int}

Realign a template hit to produce an accurate query→template index mapping.
Returns a Dict{Int,Int} mapping 1-based query positions to 1-based template positions.
"""
function realign_template(query_sequence::String, template_sequence::String,
                           hit_alignment::String;
                           hmmalign_binary::String="hmmalign",
                           hmmbuild_binary::String="hmmbuild")::Dict{Int,Int}
    if !isempty(hmmalign_binary) && check_binary_exists(hmmalign_binary)
        try
            return _realign_with_hmmalign(query_sequence, template_sequence,
                                           hmmalign_binary, hmmbuild_binary)
        catch e
            @warn "hmmalign realignment failed: $e. Falling back to Smith-Waterman."
        end
    end
    return _smith_waterman_align(query_sequence, template_sequence)
end

function _realign_with_hmmalign(query_seq::String, template_seq::String,
                                  hmmalign_binary::String,
                                  hmmbuild_binary::String)::Dict{Int,Int}
    # Build HMM from query
    hmmbuild_cfg = HmmbuildConfig(binary_path=hmmbuild_binary, n_cpu=1)
    hmm = run_hmmbuild(hmmbuild_cfg, ">query\n$query_seq\n")

    # Align template against query HMM
    hmmalign_cfg = HmmalignConfig(binary_path=hmmalign_binary)
    seqs_fasta = ">query\n$query_seq\n>template\n$template_seq\n"
    aligned = run_hmmalign(hmmalign_cfg, hmm, seqs_fasta)

    # Parse aligned output
    records = parse_fasta(aligned)
    length(records) < 2 && return _smith_waterman_align(query_seq, template_seq)

    q_aligned = records[1].sequence
    t_aligned = records[2].sequence

    return _mapping_from_aligned_pair(q_aligned, t_aligned)
end

function _mapping_from_aligned_pair(q_aligned::String, t_aligned::String)::Dict{Int,Int}
    mapping = Dict{Int,Int}()
    q_idx = 0
    t_idx = 0
    for (qc, tc) in zip(q_aligned, t_aligned)
        qgap = qc == '-'
        tgap = tc == '-'
        qgap || (q_idx += 1)
        tgap || (t_idx += 1)
        (!qgap && !tgap) && (mapping[q_idx] = t_idx)
    end
    return mapping
end

"""
    _smith_waterman_align(seq1::String, seq2::String) -> Dict{Int,Int}

Simple Smith-Waterman local alignment fallback.
Returns a query→target position mapping.
"""
function _smith_waterman_align(seq1::String, seq2::String)::Dict{Int,Int}
    n, m = length(seq1), length(seq2)
    GAP_PENALTY = -1f0
    MATCH = 2f0
    MISMATCH = -1f0

    # Score matrix
    H = zeros(Float32, n+1, m+1)
    for i in 1:n, j in 1:m
        match = seq1[i] == seq2[j] ? MATCH : MISMATCH
        H[i+1,j+1] = max(0f0,
            H[i,j]   + match,
            H[i,j+1] + GAP_PENALTY,
            H[i+1,j] + GAP_PENALTY,
        )
    end

    # Traceback from maximum
    max_val, max_pos = findmax(H)
    max_val <= 0 && return Dict{Int,Int}()

    i, j = max_pos[1], max_pos[2]
    mapping = Dict{Int,Int}()
    while i > 1 && j > 1 && H[i,j] > 0
        di, dj = i-1, j-1
        match = seq1[di] == seq2[dj] ? MATCH : MISMATCH
        if H[i,j] ≈ H[di,dj] + match
            mapping[di] = dj
            i, j = di, dj
        elseif H[i,j] ≈ H[di,j] + GAP_PENALTY
            i -= 1
        else
            j -= 1
        end
    end
    return mapping
end

"""
    parse_hmmsearch_output(tblout::String) -> Vector{Dict{String,Any}}

Parse hmmsearch tblout format into hit records.
"""
function parse_hmmsearch_output(tblout::String)::Vector{Dict{String,Any}}
    return parse_hmmsearch_tblout(tblout)
end
