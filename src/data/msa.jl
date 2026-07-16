"""
msa.jl — MSA data structure and featurization.
"""

using Logging

# ──────────────────────────────────────────────────────────────────────────────
# Msa struct
# ──────────────────────────────────────────────────────────────────────────────

"""
    Msa

Multiple sequence alignment data.
"""
struct Msa
    sequences::Vector{String}          # aligned sequences (gaps but no insertions)
    descriptions::Vector{String}
    deletion_matrix::Matrix{Int}       # (n_seqs, alignment_length)
    query_sequence::String
    chain_poly_type::String
end

function n_seqs(msa::Msa)::Int
    return length(msa.sequences)
end

function alignment_length(msa::Msa)::Int
    isempty(msa.sequences) && return 0
    return length(msa.sequences[1])
end

function Base.show(io::IO, m::Msa)
    print(io, "Msa($(n_seqs(m)) seqs, len=$(alignment_length(m)), type=$(m.chain_poly_type))")
end

# ──────────────────────────────────────────────────────────────────────────────
# Constructors
# ──────────────────────────────────────────────────────────────────────────────

"""
    Msa.from_a3m(query_sequence, chain_poly_type, a3m; deduplicate=false) -> Msa

Parse A3M text into an Msa struct.
"""
function msa_from_a3m(query_sequence::String, chain_poly_type::String,
                       a3m::String; deduplicate::Bool=false)::Msa
    descs, seqs_raw = parse_a3m(a3m)
    isempty(seqs_raw) && return msa_from_empty(query_sequence, chain_poly_type)

    # Compute deletion matrix and aligned sequences
    aligned_seqs = a3m_to_aligned_sequences(seqs_raw)
    del_mat = compute_deletion_matrix(seqs_raw)

    if deduplicate
        seen = Set{String}()
        keep = Int[]
        for i in 1:length(aligned_seqs)
            s = aligned_seqs[i]
            s ∉ seen && (push!(seen, s); push!(keep, i))
        end
        aligned_seqs = aligned_seqs[keep]
        descs        = descs[keep]
        del_mat      = del_mat[keep, :]
    end

    return Msa(aligned_seqs, descs, del_mat, query_sequence, chain_poly_type)
end

"""
    msa_from_empty(query_sequence, chain_poly_type) -> Msa

Create an Msa containing only the query sequence.
"""
function msa_from_empty(query_sequence::String, chain_poly_type::String)::Msa
    return Msa(
        [query_sequence],
        ["query"],
        zeros(Int, 1, length(query_sequence)),
        query_sequence,
        chain_poly_type,
    )
end

"""
    msa_from_multiple(msas::Vector{Msa}; deduplicate=false) -> Msa

Concatenate and optionally deduplicate multiple Msa objects.
All msas must have the same query_sequence and chain_poly_type.
"""
function msa_from_multiple(msas::Vector{Msa}; deduplicate::Bool=false)::Msa
    isempty(msas) && error("Cannot merge empty list of Msas")
    length(msas) == 1 && return msas[1]

    q = msas[1].query_sequence
    ct = msas[1].chain_poly_type
    all_seqs  = vcat([m.sequences for m in msas]...)
    all_descs = vcat([m.descriptions for m in msas]...)
    all_del   = vcat([m.deletion_matrix for m in msas]...)

    if deduplicate
        seen = Set{String}()
        keep = Int[]
        for i in 1:length(all_seqs)
            all_seqs[i] ∉ seen && (push!(seen, all_seqs[i]); push!(keep, i))
        end
        all_seqs  = all_seqs[keep]
        all_descs = all_descs[keep]
        all_del   = all_del[keep, :]
    end

    return Msa(all_seqs, all_descs, all_del, q, ct)
end

# ──────────────────────────────────────────────────────────────────────────────
# Serialization
# ──────────────────────────────────────────────────────────────────────────────

"""
    to_a3m(msa::Msa) -> String

Serialize an Msa to A3M format.
"""
function to_a3m(msa::Msa)::String
    buf = IOBuffer()
    for (desc, seq) in zip(msa.descriptions, msa.sequences)
        println(buf, ">$desc")
        println(buf, seq)
    end
    return String(take!(buf))
end

# ──────────────────────────────────────────────────────────────────────────────
# Featurization
# ──────────────────────────────────────────────────────────────────────────────

"""
    featurize(msa::Msa) -> Dict{String,Array}

Encode MSA as integer arrays for model input.
Returns "msa" (Int8), "deletion_matrix" (Int8), "msa_mask" (Bool).
"""
function featurize(msa::Msa)::Dict{String,Array}
    n = n_seqs(msa)
    L = alignment_length(msa)

    # Integer encode residues
    msa_encoded = zeros(Int8, n, L)
    mask = ones(Bool, n, L)
    for i in 1:n
        seq = msa.sequences[i]
        for j in 1:min(L, length(seq))
            c = string(seq[j])
            idx = get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, c, 0)
            msa_encoded[i, j] = Int8(min(idx, 127))
            mask[i, j] = (c != "-")
        end
    end

    del_matrix = Int8.(clamp.(msa.deletion_matrix, -128, 127))

    return Dict{String,Array}(
        "msa"             => msa_encoded,
        "deletion_matrix" => del_matrix,
        "msa_mask"        => mask,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# MSA profile computation
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_msa_profile(msa::Msa) -> Tuple{Matrix{Float32}, Vector{Float32}}

Compute residue frequency profile and mean deletion counts.
Returns (profile, deletion_mean) where:
- profile: (alignment_length, num_polymer_types+1) Float32 matrix
- deletion_mean: (alignment_length,) Float32 vector
"""
function compute_msa_profile(msa::Msa)::Tuple{Matrix{Float32},Vector{Float32}}
    n = n_seqs(msa)
    L = alignment_length(msa)
    n_types = POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP

    profile = zeros(Float32, L, n_types + 1)
    del_mean = zeros(Float32, L)

    n == 0 && return profile, del_mean

    for i in 1:n
        seq = msa.sequences[i]
        for j in 1:min(L, length(seq))
            c = string(seq[j])
            idx = get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, c, 0)
            if idx > 0
                profile[j, idx] += 1f0
            end
        end
        if size(msa.deletion_matrix, 2) >= L
            del_mean .+= Float32.(msa.deletion_matrix[i, 1:L])
        end
    end

    # Normalize
    row_sums = sum(profile, dims=2)
    for j in 1:L
        s = row_sums[j, 1]
        if s > 0
            profile[j, :] ./= s
        end
    end
    del_mean ./= n

    return profile, del_mean
end

"""
    truncate_msa(msa::Msa, max_seqs::Int) -> Msa

Truncate MSA to at most max_seqs sequences (keeping query first).
"""
function truncate_msa(msa::Msa, max_seqs::Int)::Msa
    n = n_seqs(msa)
    n <= max_seqs && return msa
    keep = 1:max_seqs
    return Msa(
        msa.sequences[keep],
        msa.descriptions[keep],
        msa.deletion_matrix[keep, :],
        msa.query_sequence,
        msa.chain_poly_type,
    )
end

"""
    deduplicate_unpaired_against_paired(unpaired::Msa, paired::Msa) -> Msa

Remove sequences from unpaired MSA that appear in the paired MSA.
"""
function deduplicate_unpaired_against_paired(unpaired::Msa, paired::Msa)::Msa
    n_seqs(unpaired) == 0 || n_seqs(paired) == 0 && return unpaired
    paired_seqs = Set(paired.sequences)
    keep = Int[]
    for i in 1:n_seqs(unpaired)
        unpaired.sequences[i] ∉ paired_seqs && push!(keep, i)
    end
    isempty(keep) && return msa_from_empty(unpaired.query_sequence, unpaired.chain_poly_type)
    return Msa(
        unpaired.sequences[keep],
        unpaired.descriptions[keep],
        unpaired.deletion_matrix[keep, :],
        unpaired.query_sequence,
        unpaired.chain_poly_type,
    )
end
