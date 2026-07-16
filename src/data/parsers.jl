"""
parsers.jl — MSA format parsers (FASTA, A3M, Stockholm).
"""

# Re-export parsers from src/parsers/
# These functions extend the base parsers with MSA-specific logic.

"""
    parse_a3m(a3m_text::String) -> Tuple{Vector{String}, Vector{String}}

Parse an A3M format string, returning (descriptions, sequences).
Sequences retain lowercase insertions.
"""
function parse_a3m(a3m_text::String)::Tuple{Vector{String},Vector{String}}
    descs = String[]
    seqs  = String[]
    current_desc = nothing
    seq_buf = IOBuffer()

    for line in eachline(IOBuffer(a3m_text))
        stripped = strip(line)
        isempty(stripped) && continue
        if startswith(stripped, '>')
            if current_desc !== nothing
                push!(descs, current_desc)
                push!(seqs, String(take!(seq_buf)))
                seq_buf = IOBuffer()
            end
            current_desc = strip(stripped[2:end])
        else
            print(seq_buf, stripped)
        end
    end
    if current_desc !== nothing
        push!(descs, current_desc)
        push!(seqs, String(take!(seq_buf)))
    end
    return descs, seqs
end

"""
    convert_a3m_to_fasta(a3m_text::String) -> String

Convert A3M to gapped FASTA by stripping lowercase insertions.
"""
function convert_a3m_to_fasta(a3m_text::String)::String
    descs, seqs = parse_a3m(a3m_text)
    buf = IOBuffer()
    for (d, s) in zip(descs, seqs)
        println(buf, ">$d")
        println(buf, filter(!islowercase, s))
    end
    return String(take!(buf))
end

"""
    a3m_sequence_to_aligned(seq::String) -> String

Convert an A3M sequence (with lowercase insertions) to an aligned sequence
(uppercase only, with gaps preserved).
"""
function a3m_sequence_to_aligned(seq::String)::String
    return filter(c -> !islowercase(c), seq)
end

"""
    parse_stockholm(sto_text::String) -> Vector{Tuple{String,String}}

Parse Stockholm format alignment. Returns (name, sequence) pairs in order.
"""
function parse_stockholm(sto_text::String)::Vector{Tuple{String,String}}
    sequences = OrderedDict{String,IOBuffer}()
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

    return [(name, String(take!(sequences[name]))) for name in order]
end

# We need OrderedDict
const OrderedDict = Dict  # Use regular Dict for now; keys preserved by insertion order in Julia 1.7+

"""
    parse_jackhmmer_a3m_output(sto_text::String) -> String

Convert jackhmmer Stockholm output to A3M format.
"""
function parse_jackhmmer_a3m_output(sto_text::String)::String
    return stockholm_to_a3m(sto_text)
end
