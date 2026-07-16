"""
msa_conversion.jl — MSA format conversion utilities (A3M ↔ FASTA ↔ Stockholm).
"""

# ──────────────────────────────────────────────────────────────────────────────
# A3M → aligned FASTA
# ──────────────────────────────────────────────────────────────────────────────

"""
    a3m_to_fasta(a3m_text::String) -> Vector{Tuple{String,String}}

Convert an A3M multiple sequence alignment to FASTA format.
In A3M: uppercase = aligned, lowercase = insertions (not in query columns).
This function strips lowercase characters to produce gapped FASTA.
"""
function a3m_to_fasta(a3m_text::String)::Vector{Tuple{String,String}}
    records = Tuple{String,String}[]
    current_desc = nothing
    seq_buf = IOBuffer()

    for line in eachline(IOBuffer(a3m_text))
        stripped = strip(line)
        isempty(stripped) && continue
        if startswith(stripped, '>')
            if current_desc !== nothing
                push!(records, (current_desc, String(take!(seq_buf))))
                seq_buf = IOBuffer()
            end
            current_desc = strip(stripped[2:end])
        else
            # Remove lowercase (insertion) characters
            for ch in stripped
                islowercase(ch) || print(seq_buf, ch)
            end
        end
    end
    if current_desc !== nothing
        push!(records, (current_desc, String(take!(seq_buf))))
    end
    return records
end

"""
    fasta_to_a3m(records::Vector{Tuple{String,String}}) -> String

Convert gapped FASTA records to A3M format.
This just outputs FASTA (A3M is FASTA when no insertions are present).
"""
function fasta_to_a3m(records::Vector{Tuple{String,String}})::String
    buf = IOBuffer()
    for (desc, seq) in records
        println(buf, ">$desc")
        println(buf, seq)
    end
    return String(take!(buf))
end

# ──────────────────────────────────────────────────────────────────────────────
# Stockholm → A3M
# ──────────────────────────────────────────────────────────────────────────────

"""
    stockholm_to_a3m(sto_text::String) -> String

Convert Stockholm format MSA to A3M format.
Stockholm is the native output format of jackhmmer/hmmbuild.
"""
function stockholm_to_a3m(sto_text::String)::String
    sequences = Dict{String,IOBuffer}()
    order = String[]

    for line in eachline(IOBuffer(sto_text))
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, '#') && continue
        stripped == "//" && continue

        parts = split(stripped, r"\s+", limit=2)
        length(parts) != 2 && continue
        name, seq = parts[1], parts[2]

        if !haskey(sequences, name)
            sequences[name] = IOBuffer()
            push!(order, name)
        end
        print(sequences[name], seq)
    end

    # Convert to A3M: determine which columns are query (uppercase, non-gap)
    # In Stockholm: all columns are aligned. Map '.' → '-', keep case.
    buf = IOBuffer()
    for name in order
        seq = String(take!(sequences[name]))
        a3m_seq = replace(seq, '.' => '-')
        println(buf, ">$name")
        println(buf, a3m_seq)
    end
    return String(take!(buf))
end

# ──────────────────────────────────────────────────────────────────────────────
# Deletion matrix computation
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_deletion_matrix(a3m_sequences::Vector{String}) -> Matrix{Int}

Compute deletion matrix from A3M sequences (not the gapped FASTA form).
The deletion matrix has shape (num_sequences, query_length).
Entry [i,j] = number of lowercase (insertion) characters before position j in sequence i.
"""
function compute_deletion_matrix(a3m_sequences::Vector{String})::Matrix{Int}
    if isempty(a3m_sequences)
        return zeros(Int, 0, 0)
    end
    # Compute query length = number of non-lowercase chars in query
    query_len = count(ch -> !islowercase(ch) && ch != '-', a3m_sequences[1])
    query_len == 0 && return zeros(Int, length(a3m_sequences), 1)

    n = length(a3m_sequences)
    mat = zeros(Int, n, query_len)

    for (i, seq) in enumerate(a3m_sequences)
        col = 0
        del_count = 0
        for ch in seq
            if islowercase(ch)
                del_count += 1
            else
                col += 1
                col > query_len && break
                mat[i, col] = del_count
                del_count = 0
            end
        end
    end
    return mat
end

"""
    a3m_to_aligned_sequences(a3m_sequences::Vector{String}) -> Vector{String}

Strip lowercase (insertion) characters from A3M sequences to produce aligned sequences.
"""
function a3m_to_aligned_sequences(a3m_sequences::Vector{String})::Vector{String}
    return [String(filter(ch -> !islowercase(ch), seq)) for seq in a3m_sequences]
end

# ──────────────────────────────────────────────────────────────────────────────
# A3M query sequence extraction
# ──────────────────────────────────────────────────────────────────────────────

"""
    get_query_sequence(a3m_text::String) -> String

Extract the first (query) sequence from an A3M string.
Lowercase characters are stripped, gaps removed.
"""
function get_query_sequence(a3m_text::String)::String
    for line in eachline(IOBuffer(a3m_text))
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, '>') && continue
        # First sequence line found
        return String(filter(ch -> !islowercase(ch) && ch != '-', stripped))
    end
    return ""
end
