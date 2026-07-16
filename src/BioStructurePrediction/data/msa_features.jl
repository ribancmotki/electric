"""
Conversion of MSA data into model input feature arrays.
"""

# Amino acid one-hot encoding: 20 amino acids + gaps + unknown
const MSA_AA_ORDER = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-','X']
const MSA_AA_TO_IDX = Dict(c => i for (i,c) in enumerate(MSA_AA_ORDER))
const NUM_MSA_CHANNELS = 34  # 22 one-hot + 1 gap + 1 deletion indicator + ... see model spec

# Full 34-channel MSA feature encoding (matches model spec)
# Channels 1-21: one-hot over 20 AA + gap
# Channel 22: has-deletion (0/1)
# Channel 23: deletion value (log-scaled)
# Channel 24: deletion mean (log-scaled over column)
# Channel 25-34: profile (10 bins)
const NUM_MSA_ONEHOT = 22  # 20 AA + gap + unknown
const NUM_EXTRA_MSA_CHANNELS = 25

"""
    make_msa_features(msa::Msa) -> Dict{String,Array}

Convert an Msa to the feature arrays required by the model.

Returns:
- "msa": Int32 (n_seqs, aln_len) — residue index per position
- "msa_mask": Float32 (n_seqs, aln_len) — 1.0 for non-gap positions
- "deletion_matrix_int": Int32 (n_seqs, aln_len)
- "deletion_matrix": Float32 (n_seqs, aln_len) — arctan-normalised deletion counts
- "msa_feat": Float32 (n_seqs, aln_len, NUM_MSA_CHANNELS) — per-position features
- "num_alignments": Int32 scalar
"""
function make_msa_features(msa::Msa)::Dict{String,Array}
    n = n_seqs(msa)
    aln_len = alignment_length(msa)

    if n == 0 || aln_len == 0
        return Dict{String,Array}(
            "msa"               => zeros(Int32, 1, 0),
            "msa_mask"          => zeros(Float32, 1, 0),
            "deletion_matrix_int" => zeros(Int32, 1, 0),
            "deletion_matrix"   => zeros(Float32, 1, 0),
            "msa_feat"          => zeros(Float32, 1, 0, NUM_MSA_CHANNELS),
            "num_alignments"    => Int32[0],
        )
    end

    # Build one-hot residue array
    msa_aa = zeros(Int32, n, aln_len)
    for i in 1:n
        for j in 1:aln_len
            c = j <= length(msa.sequences[i]) ? msa.sequences[i][j] : '-'
            msa_aa[i, j] = Int32(get(MSA_AA_TO_IDX, c, get(MSA_AA_TO_IDX, 'X', 22)))
        end
    end

    # Mask: 1.0 where not gap
    msa_mask = Float32.(msa_aa .!= get(MSA_AA_TO_IDX, '-', 21))

    # Deletion matrix
    del_mat_int = size(msa.deletion_matrix) == (n, aln_len) ?
        Int32.(msa.deletion_matrix) : zeros(Int32, n, aln_len)
    del_mat = Float32.(atan.(del_mat_int ./ 3f0) .* (2f0 / π))

    # Build full MSA feature tensor: (n, aln_len, 34)
    msa_feat = zeros(Float32, n, aln_len, NUM_MSA_CHANNELS)
    for i in 1:n
        for j in 1:aln_len
            aa_idx = msa_aa[i, j]
            # Channels 1-22: one-hot over amino acids + gap
            if 1 <= aa_idx <= NUM_MSA_ONEHOT
                msa_feat[i, j, aa_idx] = 1f0
            else
                msa_feat[i, j, NUM_MSA_ONEHOT] = 1f0  # unknown → last one-hot
            end
            # Channel 23: has-deletion indicator
            msa_feat[i, j, 23] = del_mat_int[i, j] > 0 ? 1f0 : 0f0
            # Channel 24: normalised deletion value
            msa_feat[i, j, 24] = del_mat[i, j]
        end
    end

    # Compute per-column profile (fraction of each amino acid)
    profile = zeros(Float32, aln_len, NUM_MSA_ONEHOT)
    for j in 1:aln_len
        col_counts = zeros(Float32, NUM_MSA_ONEHOT)
        n_valid = 0f0
        for i in 1:n
            aa_idx = msa_aa[i, j]
            if 1 <= aa_idx <= NUM_MSA_ONEHOT
                col_counts[aa_idx] += 1f0
                n_valid += 1f0
            end
        end
        n_valid > 0 && (profile[j, :] = col_counts ./ n_valid)
    end
    # Channels 25-34: truncated profile (first 10 dims if NUM_MSA_ONEHOT >= 10)
    profile_dims = min(10, NUM_MSA_ONEHOT)
    for i in 1:n
        for j in 1:aln_len
            msa_feat[i, j, 25:24+profile_dims] = profile[j, 1:profile_dims]
        end
    end

    return Dict{String,Array}(
        "msa"                 => msa_aa,
        "msa_mask"            => msa_mask,
        "deletion_matrix_int" => del_mat_int,
        "deletion_matrix"     => del_mat,
        "msa_feat"            => msa_feat,
        "num_alignments"      => Int32[n],
    )
end

"""
    make_extra_msa_features(extra_msa::Msa) -> Dict{String,Array}

Create extra MSA features (25 channels, lower quality sequences).
"""
function make_extra_msa_features(extra_msa::Msa)::Dict{String,Array}
    n = n_seqs(extra_msa)
    aln_len = alignment_length(extra_msa)

    if n == 0 || aln_len == 0
        return Dict{String,Array}(
            "extra_msa"         => zeros(Int32, 1, 0),
            "extra_msa_mask"    => zeros(Float32, 1, 0),
            "extra_msa_feat"    => zeros(Float32, 1, 0, NUM_EXTRA_MSA_CHANNELS),
            "extra_deletion_matrix" => zeros(Float32, 1, 0),
        )
    end

    extra_msa_aa = zeros(Int32, n, aln_len)
    for i in 1:n
        for j in 1:aln_len
            c = j <= length(extra_msa.sequences[i]) ? extra_msa.sequences[i][j] : '-'
            extra_msa_aa[i, j] = Int32(get(MSA_AA_TO_IDX, c, 22))
        end
    end

    extra_msa_mask = Float32.(extra_msa_aa .!= get(MSA_AA_TO_IDX, '-', 21))

    del_mat_int = size(extra_msa.deletion_matrix) == (n, aln_len) ?
        Int32.(extra_msa.deletion_matrix) : zeros(Int32, n, aln_len)
    del_mat = Float32.(atan.(del_mat_int ./ 3f0) .* (2f0 / π))

    # 25-channel extra MSA features
    extra_msa_feat = zeros(Float32, n, aln_len, NUM_EXTRA_MSA_CHANNELS)
    for i in 1:n
        for j in 1:aln_len
            aa_idx = extra_msa_aa[i, j]
            if 1 <= aa_idx <= NUM_MSA_ONEHOT
                extra_msa_feat[i, j, aa_idx] = 1f0
            else
                extra_msa_feat[i, j, NUM_MSA_ONEHOT] = 1f0
            end
            extra_msa_feat[i, j, 23] = del_mat_int[i, j] > 0 ? 1f0 : 0f0
            extra_msa_feat[i, j, 24] = del_mat[i, j]
        end
    end

    return Dict{String,Array}(
        "extra_msa"             => extra_msa_aa,
        "extra_msa_mask"        => extra_msa_mask,
        "extra_msa_feat"        => extra_msa_feat,
        "extra_deletion_matrix" => del_mat,
    )
end
