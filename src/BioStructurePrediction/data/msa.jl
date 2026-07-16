"""
Multiple Sequence Alignment (MSA) data model and operations.
"""

# ──────────────────────────────────────────────
#  Msa struct
# ──────────────────────────────────────────────

"""
    Msa

Represents a multiple sequence alignment.

Fields:
- sequences: aligned sequences (uppercase with gap characters -)
- descriptions: sequence description/header lines
- deletion_matrix: (n_seqs × alignment_length) integer matrix; entry [i,j] = number of
  insertions before position j in sequence i (from A3M encoding)
"""
struct Msa
    sequences::Vector{String}
    descriptions::Vector{String}
    deletion_matrix::Matrix{Int}

    function Msa(sequences::Vector{String}, descriptions::Vector{String}, deletion_matrix::Matrix{Int})
        length(sequences) == length(descriptions) ||
            error("sequences and descriptions length mismatch")
        n = length(sequences)
        if n > 0
            aln_len = length(first(sequences))
            size(deletion_matrix) == (n, aln_len) ||
                error("deletion_matrix shape $(size(deletion_matrix)) != ($n, $aln_len)")
        end
        new(sequences, descriptions, deletion_matrix)
    end
end

"""
    Msa(sequences, descriptions) -> Msa

Construct an Msa with a zero deletion matrix.
"""
function Msa(sequences::Vector{String}, descriptions::Vector{String})::Msa
    n = length(sequences)
    aln_len = n > 0 ? length(first(sequences)) : 0
    return Msa(sequences, descriptions, zeros(Int, n, aln_len))
end

"""
    Msa_from_a3m(text::String) -> Msa

Parse an A3M string into an Msa object.
"""
function Msa_from_a3m(text::String)::Msa
    raw_records = parse_a3m(text)
    descriptions = String[r[1] for r in raw_records]
    aligned_seqs = String[]
    deletion_rows = Vector{Int}[]
    for (_, seq) in raw_records
        aln_seq, del_row = a3m_sequence_to_aligned(seq)
        push!(aligned_seqs, aln_seq)
        push!(deletion_rows, del_row)
    end
    aln_len = isempty(aligned_seqs) ? 0 : length(first(aligned_seqs))
    deletion_matrix = Matrix{Int}(undef, length(aligned_seqs), aln_len)
    for (i, row) in enumerate(deletion_rows)
        if length(row) != aln_len
            # Pad or truncate to alignment length
            row_padded = vcat(row, zeros(Int, max(0, aln_len - length(row))))
            deletion_matrix[i, :] = row_padded[1:aln_len]
        else
            deletion_matrix[i, :] = row
        end
    end
    return Msa(aligned_seqs, descriptions, deletion_matrix)
end

"""
    Msa_from_stockholm(text::String) -> Msa

Parse a Stockholm string into an Msa object.
"""
function Msa_from_stockholm(text::String)::Msa
    records = parse_stockholm(text)
    isempty(records) && return Msa(String[], String[], zeros(Int, 0, 0))
    descriptions = String[r[1] for r in records]
    sequences    = String[r[2] for r in records]
    n = length(sequences)
    aln_len = length(first(sequences))
    deletion_matrix = zeros(Int, n, aln_len)
    return Msa(sequences, descriptions, deletion_matrix)
end

"""
    Msa_from_fasta(text::String) -> Msa

Parse a FASTA string into an Msa object.
"""
function Msa_from_fasta(text::String)::Msa
    records = parse_fasta(text)
    isempty(records) && return Msa(String[], String[], zeros(Int, 0, 0))
    descriptions = String[r[1] for r in records]
    sequences    = String[r[2] for r in records]
    n = length(sequences)
    aln_len = length(first(sequences))
    deletion_matrix = zeros(Int, n, aln_len)
    return Msa(sequences, descriptions, deletion_matrix)
end

"""
    msa_to_a3m(msa::Msa) -> String

Serialise an Msa back to A3M format.
"""
function msa_to_a3m(msa::Msa)::String
    buf = IOBuffer()
    for (i, (desc, seq)) in enumerate(zip(msa.descriptions, msa.sequences))
        println(buf, ">$desc")
        # Re-encode insertions as lowercase
        aln_len = length(seq)
        if aln_len > 0 && size(msa.deletion_matrix, 1) >= i
            row = msa.deletion_matrix[i, :]
            encoded = IOBuffer()
            for (j, c) in enumerate(seq)
                if j <= length(row) && row[j] > 0
                    print(encoded, 'x'^row[j])  # lowercase placeholder
                end
                print(encoded, c)
            end
            println(buf, String(take!(encoded)))
        else
            println(buf, seq)
        end
    end
    return String(take!(buf))
end

# ──────────────────────────────────────────────
#  MSA operations
# ──────────────────────────────────────────────

"""
    n_seqs(msa::Msa) -> Int

Return the number of sequences in the alignment.
"""
function n_seqs(msa::Msa)::Int
    return length(msa.sequences)
end

"""
    alignment_length(msa::Msa) -> Int

Return the alignment column length.
"""
function alignment_length(msa::Msa)::Int
    return isempty(msa.sequences) ? 0 : length(first(msa.sequences))
end

"""
    truncate_msa(msa::Msa, max_seqs::Int) -> Msa

Return the first max_seqs sequences. The query sequence (index 1) is always preserved.
"""
function truncate_msa(msa::Msa, max_seqs::Int)::Msa
    n = n_seqs(msa)
    max_seqs >= n && return msa
    seqs  = msa.sequences[1:max_seqs]
    descs = msa.descriptions[1:max_seqs]
    dmat  = msa.deletion_matrix[1:max_seqs, :]
    return Msa(seqs, descs, dmat)
end

"""
    deduplicate_unpaired_against_paired(unpaired::Msa, paired::Msa) -> Msa

Remove from `unpaired` any sequence that appears in `paired` (exact match after uppercasing and removing gaps).
The query (index 1 of unpaired) is always preserved.
"""
function deduplicate_unpaired_against_paired(unpaired::Msa, paired::Msa)::Msa
    if n_seqs(unpaired) == 0 || n_seqs(paired) == 0
        return unpaired
    end

    # Build set of sequences in paired MSA (stripped of gaps)
    paired_seqs = Set{String}()
    for seq in paired.sequences
        push!(paired_seqs, replace(seq, '-' => ""))
    end

    keep = Bool[true for _ in 1:n_seqs(unpaired)]
    for i in 2:n_seqs(unpaired)
        stripped = replace(unpaired.sequences[i], '-' => "")
        if stripped in paired_seqs
            keep[i] = false
        end
    end

    if all(keep)
        return unpaired
    end

    idxs = findall(keep)
    return Msa(
        unpaired.sequences[idxs],
        unpaired.descriptions[idxs],
        unpaired.deletion_matrix[idxs, :],
    )
end

"""
    merge_msas(msas::Vector{Msa}) -> Msa

Concatenate multiple MSAs, deduplicating by sequence content (after stripping gaps).
The query sequence from the first MSA is always at position 1.
"""
function merge_msas(msas::Vector{Msa})::Msa
    isempty(msas) && return Msa(String[], String[], zeros(Int, 0, 0))
    seen_seqs  = Set{String}()
    all_seqs   = String[]
    all_descs  = String[]
    all_del_rows = Vector{Int}[]

    for msa in msas
        aln_len = alignment_length(msa)
        for i in 1:n_seqs(msa)
            stripped = replace(msa.sequences[i], '-' => "")
            if stripped ∉ seen_seqs
                push!(seen_seqs, stripped)
                push!(all_seqs, msa.sequences[i])
                push!(all_descs, msa.descriptions[i])
                push!(all_del_rows, size(msa.deletion_matrix, 1) >= i ?
                    msa.deletion_matrix[i, :] : zeros(Int, aln_len))
            end
        end
    end

    n = length(all_seqs)
    aln_len = n > 0 ? length(first(all_seqs)) : 0
    del_mat = Matrix{Int}(undef, n, aln_len)
    for (i, row) in enumerate(all_del_rows)
        len_r = length(row)
        if len_r >= aln_len
            del_mat[i, :] = row[1:aln_len]
        else
            del_mat[i, 1:len_r]           = row
            del_mat[i, len_r+1:aln_len]   = zeros(Int, aln_len - len_r)
        end
    end

    return Msa(all_seqs, all_descs, del_mat)
end
