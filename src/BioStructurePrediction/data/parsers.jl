"""
MSA file format parsers: A3M, Stockholm, FASTA.
"""

# ──────────────────────────────────────────────
#  FASTA Parser
# ──────────────────────────────────────────────

"""
    parse_fasta(text::String) -> Vector{Tuple{String,String}}

Parse a FASTA-format string into a list of (description, sequence) pairs.
"""
function parse_fasta(text::String)::Vector{Tuple{String,String}}
    records = Tuple{String,String}[]
    current_desc = nothing
    seq_lines    = String[]

    for line in split(text, '\n')
        line = rstrip(line)
        if startswith(line, '>')
            if current_desc !== nothing
                push!(records, (current_desc, join(seq_lines)))
            end
            current_desc = lstrip(line[2:end])
            seq_lines    = String[]
        elseif !isempty(line) && !startswith(line, ';')
            push!(seq_lines, line)
        end
    end
    if current_desc !== nothing
        push!(records, (current_desc, join(seq_lines)))
    end
    return records
end

# ──────────────────────────────────────────────
#  A3M Parser
# ──────────────────────────────────────────────

"""
    parse_a3m(text::String) -> Vector{Tuple{String,String}}

Parse an A3M-format string into a list of (description, sequence) pairs.
Insertion characters (lowercase letters) are preserved in the sequences.
"""
function parse_a3m(text::String)::Vector{Tuple{String,String}}
    return parse_fasta(text)  # A3M uses FASTA format; insertions are lowercase
end

"""
    convert_a3m_to_fasta(a3m::String) -> String

Convert an A3M string to FASTA by removing lowercase insertion characters.
"""
function convert_a3m_to_fasta(a3m::String)::String
    records = parse_a3m(a3m)
    buf = IOBuffer()
    for (desc, seq) in records
        println(buf, ">$desc")
        # Remove lowercase (insertion) characters, keep uppercase and gaps
        cleaned = filter(c -> isuppercase(c) || c == '-', seq)
        println(buf, cleaned)
    end
    return String(take!(buf))
end

"""
    a3m_sequence_to_aligned(seq::String) -> Tuple{String,Vector{Int}}

Convert an A3M sequence string to:
- aligned sequence (uppercase only, insertions removed, gaps preserved)
- deletion_matrix row: number of inserted residues before each alignment position

Returns (aligned_seq, deletion_counts).
"""
function a3m_sequence_to_aligned(seq::String)::Tuple{String,Vector{Int}}
    aligned_chars = Char[]
    deletions     = Int[]
    insertion_count = 0
    for c in seq
        if isuppercase(c) || c == '-'
            push!(deletions, insertion_count)
            push!(aligned_chars, c)
            insertion_count = 0
        elseif islowercase(c)
            insertion_count += 1
        end
    end
    return join(aligned_chars), deletions
end

# ──────────────────────────────────────────────
#  Stockholm Parser
# ──────────────────────────────────────────────

"""
    parse_stockholm(text::String) -> Vector{Tuple{String,String}}

Parse a Stockholm-format alignment string into a list of (description, sequence) pairs.
"""
function parse_stockholm(text::String)::Vector{Tuple{String,String}}
    seqs = Dict{String,IOBuffer}()
    order = String[]

    for line in split(text, '\n')
        line = rstrip(line)
        isempty(line) && continue
        startswith(line, "# STOCKHOLM") && continue
        startswith(line, "#=") && continue
        line == "//" && break
        startswith(line, '#') && continue

        parts = split(line; limit=2)
        length(parts) != 2 && continue
        name, seq = parts[1], parts[2]
        if !haskey(seqs, name)
            seqs[name] = IOBuffer()
            push!(order, name)
        end
        print(seqs[name], strip(seq))
    end

    return Tuple{String,String}[(name, String(take!(seqs[name]))) for name in order]
end

# ──────────────────────────────────────────────
#  Jackhmmer / Nhmmer output parsers
# ──────────────────────────────────────────────

"""
    parse_jackhmmer_a3m_output(stdout_str::String) -> String

Extract the A3M alignment block from Jackhmmer's stdout (with -A flag).
Jackhmmer writes the alignment in Stockholm format; convert to A3M.
"""
function parse_jackhmmer_a3m_output(stdout_str::String)::String
    # Jackhmmer -A writes Stockholm; convert Stockholm → A3M
    records = parse_stockholm(stdout_str)
    buf = IOBuffer()
    for (desc, seq) in records
        println(buf, ">$desc")
        println(buf, seq)
    end
    return String(take!(buf))
end

"""
    parse_nhmmer_stockholm_output(stdout_str::String) -> String

Extract the Stockholm alignment from Nhmmer's output.
"""
function parse_nhmmer_stockholm_output(stdout_str::String)::String
    # Nhmmer writes Stockholm directly
    return stdout_str
end
