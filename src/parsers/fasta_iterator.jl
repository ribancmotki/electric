"""
fasta_iterator.jl — FASTA format parser and iterator.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────

"""
FastaRecord: a single FASTA record.
"""
struct FastaRecord
    description::String  # full header line (without '>')
    sequence::String     # raw sequence (may include gaps or IUPAC codes)
end

function Base.show(io::IO, r::FastaRecord)
    seq_preview = length(r.sequence) > 30 ? r.sequence[1:30] * "…" : r.sequence
    print(io, "FastaRecord(\"$(r.description)\", \"$seq_preview\")")
end

# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

"""
    parse_fasta(text::String) -> Vector{FastaRecord}

Parse a FASTA string and return a list of FastaRecord objects.
Handles standard FASTA with optional line-wrapped sequences.
"""
function parse_fasta(text::String)::Vector{FastaRecord}
    records = FastaRecord[]
    current_desc = nothing
    seq_buf = IOBuffer()

    for line in eachline(IOBuffer(text))
        stripped = strip(line)
        isempty(stripped) && continue
        if startswith(stripped, '>')
            # Save previous record
            if current_desc !== nothing
                push!(records, FastaRecord(current_desc, String(take!(seq_buf))))
                seq_buf = IOBuffer()
            end
            current_desc = strip(stripped[2:end])
        elseif startswith(stripped, ';')
            # FASTA comment line — skip
            continue
        else
            # Sequence line: strip spaces
            print(seq_buf, replace(stripped, r"\s" => ""))
        end
    end
    # Save last record
    if current_desc !== nothing
        push!(records, FastaRecord(current_desc, String(take!(seq_buf))))
    end
    return records
end

"""
    parse_fasta_file(path::String) -> Vector{FastaRecord}

Parse a FASTA file from disk.
"""
function parse_fasta_file(path::String)::Vector{FastaRecord}
    return parse_fasta(read(path, String))
end

# ──────────────────────────────────────────────────────────────────────────────
# Iterator
# ──────────────────────────────────────────────────────────────────────────────

"""
    FastaIterator

Lazy iterator over FASTA records in a file or string, without loading the entire
file into memory.
"""
struct FastaIterator
    source::Union{String, IOBuffer}  # file path or pre-loaded text
    _is_path::Bool
end

FastaIterator(path::String) = FastaIterator(path, true)
FastaIterator(io::IOBuffer) = FastaIterator(io, false)

function Base.iterate(iter::FastaIterator, state=nothing)
    if state === nothing
        io = iter._is_path ? open(iter.source, "r") : iter.source
        state = (io=io, buf=IOBuffer(), desc=Ref{Union{String,Nothing}}(nothing),
                 should_close=iter._is_path)
    end
    io, buf, desc_ref, should_close = state.io, state.buf, state.desc, state.should_close

    while !eof(io)
        line = strip(readline(io))
        isempty(line) && continue
        startswith(line, ';') && continue

        if startswith(line, '>')
            if desc_ref[] !== nothing
                # Yield previous record
                record = FastaRecord(desc_ref[], String(take!(buf)))
                desc_ref[] = strip(line[2:end])
                return record, state
            else
                desc_ref[] = strip(line[2:end])
            end
        else
            print(buf, replace(line, r"\s" => ""))
        end
    end

    # Yield final record
    if desc_ref[] !== nothing
        record = FastaRecord(desc_ref[], String(take!(buf)))
        desc_ref[] = nothing
        should_close && close(io)
        return record, state
    end

    should_close && close(io)
    return nothing
end

Base.eltype(::FastaIterator) = FastaRecord
Base.IteratorSize(::FastaIterator) = Base.SizeUnknown()

# ──────────────────────────────────────────────────────────────────────────────
# Writer
# ──────────────────────────────────────────────────────────────────────────────

"""
    write_fasta(records::Vector{FastaRecord}; line_width=60) -> String

Serialize FASTA records to a string, wrapping sequences at `line_width` characters.
"""
function write_fasta(records::Vector{FastaRecord}; line_width::Int=60)::String
    buf = IOBuffer()
    for r in records
        println(buf, ">$(r.description)")
        seq = r.sequence
        for i in 1:line_width:length(seq)
            println(buf, seq[i:min(i+line_width-1, length(seq))])
        end
    end
    return String(take!(buf))
end

"""
    records_to_fasta_string(sequences::Vector{Tuple{String,String}}) -> String

Convert (description, sequence) pairs to a FASTA string.
"""
function records_to_fasta_string(seqs::Vector{Tuple{String,String}})::String
    return write_fasta([FastaRecord(d, s) for (d, s) in seqs])
end

"""
    make_single_record_fasta(sequence::String; description="query") -> String

Create a FASTA string with a single record.
"""
function make_single_record_fasta(sequence::String;
                                  description::String="query")::String
    return ">$description\n$sequence\n"
end
