"""
Template realignment using hmmalign.
"""

"""
    realign_template(
        query_seq::String,
        template_seq::String,
        hmmalign_binary::String;
        hmm_profile_path::Union{String,Nothing}=nothing
    ) -> Tuple{Vector{Int},Vector{Int}}

Align the query to the template sequence using hmmalign (or local alignment fallback).
Returns (query_indices, template_indices) — 0-based index arrays of aligned positions.
"""
function realign_template(
    query_seq::String,
    template_seq::String,
    hmmalign_binary::String;
    hmm_profile_path::Union{String,Nothing} = nothing
)::Tuple{Vector{Int},Vector{Int}}
    # If hmmalign_binary is available and a profile is provided, use it
    if !isempty(hmmalign_binary) && isfile(hmmalign_binary) && hmm_profile_path !== nothing && isfile(hmm_profile_path)
        return _run_hmmalign(query_seq, template_seq, hmmalign_binary, hmm_profile_path)
    end
    # Fallback: local pairwise alignment (Smith-Waterman)
    return _local_alignment(query_seq, template_seq)
end

"""
    _run_hmmalign(query_seq, template_seq, hmmalign_binary, hmm_profile_path)
        -> Tuple{Vector{Int},Vector{Int}}

Run hmmalign and parse the resulting Stockholm alignment.
"""
function _run_hmmalign(
    query_seq::String,
    template_seq::String,
    hmmalign_binary::String,
    hmm_profile_path::String
)::Tuple{Vector{Int},Vector{Int}}
    # Write sequences to temp files
    tmp_dir   = mktempdir()
    seq_file  = joinpath(tmp_dir, "seqs.fasta")
    aln_file  = joinpath(tmp_dir, "aligned.sto")

    try
        open(seq_file, "w") do io
            println(io, ">query")
            println(io, query_seq)
            println(io, ">template")
            println(io, template_seq)
        end

        cmd = `$hmmalign_binary --outformat afa --trim $hmm_profile_path $seq_file`
        result = read(cmd, String)

        # Parse FASTA output from hmmalign
        records = parse_fasta(result)
        query_aln    = ""
        template_aln = ""
        for (desc, seq) in records
            if occursin("query", desc)
                query_aln = seq
            elseif occursin("template", desc)
                template_aln = seq
            end
        end

        return _extract_alignment_indices(query_aln, template_aln)
    catch e
        @warn "hmmalign failed, falling back to local alignment: $e"
        return _local_alignment(query_seq, template_seq)
    finally
        rm(tmp_dir; recursive=true, force=true)
    end
end

"""
    _extract_alignment_indices(query_aln::String, template_aln::String)
        -> Tuple{Vector{Int},Vector{Int}}

From aligned sequences (with gap characters), extract the 0-based index arrays
of matching positions.
"""
function _extract_alignment_indices(
    query_aln::String,
    template_aln::String
)::Tuple{Vector{Int},Vector{Int}}
    query_indices    = Int[]
    template_indices = Int[]

    query_pos    = 0
    template_pos = 0

    for (qc, tc) in zip(query_aln, template_aln)
        is_q_gap = qc == '-'
        is_t_gap = tc == '-'

        if !is_q_gap && !is_t_gap
            push!(query_indices,    query_pos)
            push!(template_indices, template_pos)
        end

        is_q_gap || (query_pos    += 1)
        is_t_gap || (template_pos += 1)
    end

    return query_indices, template_indices
end

"""
    _local_alignment(query_seq::String, template_seq::String)
        -> Tuple{Vector{Int},Vector{Int}}

Simple Smith-Waterman local alignment fallback.
"""
function _local_alignment(query_seq::String, template_seq::String)::Tuple{Vector{Int},Vector{Int}}
    # Substitution: match = +2, mismatch = -1, gap = -2
    match_score    = 2
    mismatch_score = -1
    gap_open       = -2
    gap_extend     = -1

    q = collect(query_seq)
    t = collect(template_seq)
    m = length(q)
    n = length(t)

    H = zeros(Float32, m+1, n+1)
    traceback = fill('s', m+1, n+1)  # 's' = stop, 'd' = diagonal, 'l' = left, 'u' = up

    max_score = 0f0
    max_i, max_j = 1, 1

    for i in 1:m, j in 1:n
        diag_score = H[i, j]   + Float32(q[i] == t[j] ? match_score : mismatch_score)
        left_score = H[i+1, j] + Float32(gap_open)
        up_score   = H[i, j+1] + Float32(gap_open)
        best = max(0f0, diag_score, left_score, up_score)
        H[i+1, j+1] = best
        if best > max_score
            max_score = best
            max_i, max_j = i+1, j+1
        end
        if best == diag_score
            traceback[i+1, j+1] = 'd'
        elseif best == left_score
            traceback[i+1, j+1] = 'l'
        elseif best == up_score
            traceback[i+1, j+1] = 'u'
        end
    end

    # Trace back
    query_indices    = Int[]
    template_indices = Int[]
    i, j = max_i, max_j
    while i > 1 && j > 1 && H[i, j] > 0f0
        tb = traceback[i, j]
        if tb == 'd'
            push!(query_indices,    i - 2)
            push!(template_indices, j - 2)
            i -= 1; j -= 1
        elseif tb == 'l'
            j -= 1
        elseif tb == 'u'
            i -= 1
        else
            break
        end
    end

    reverse!(query_indices)
    reverse!(template_indices)
    return query_indices, template_indices
end

"""
    parse_hmmsearch_output(stdout_str::String, max_template_date::Union{Date,Nothing})
        -> Vector{NamedTuple}

Parse hmmsearch tblout output and return a list of hits sorted by E-value.
Each hit has: pdb_id, chain_id, evalue, score.
"""
function parse_hmmsearch_output(
    stdout_str::String,
    max_template_date::Union{Date,Nothing}
)::Vector{NamedTuple}
    hits = NamedTuple[]
    for line in split(stdout_str, '\n')
        startswith(line, '#') && continue
        isempty(strip(line))  && continue
        parts = split(line)
        length(parts) < 6    && continue
        target_name = String(parts[1])
        evalue      = tryparse(Float64, parts[5])
        score       = tryparse(Float64, parts[6])
        evalue === nothing || score === nothing && continue

        # Parse PDB ID and chain from target name (e.g., "1ABC_A/1-200")
        m = match(r"^([0-9A-Za-z]{4})_([A-Za-z0-9]+)", target_name)
        if m !== nothing
            pdb_id   = String(m.captures[1])
            chain_id = String(m.captures[2])
            push!(hits, (pdb_id=pdb_id, chain_id=chain_id, evalue=evalue, score=score))
        end
    end
    # Sort by E-value ascending
    sort!(hits, by = h -> h.evalue)
    return hits
end
