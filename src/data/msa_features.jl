"""
msa_features.jl — MSA featurization for model input.
"""

"""
    make_msa_features(msa::Msa) -> Dict{String,Array}

Compute 34-channel per-position MSA features.
"""
function make_msa_features(msa::Msa)::Dict{String,Array}
    n = n_seqs(msa)
    L = alignment_length(msa)
    n_types = POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP

    # Residue type one-hot: n × L × n_types
    msa_one_hot = zeros(Float32, n, L, n_types)
    for i in 1:n
        seq = msa.sequences[i]
        for j in 1:min(L, length(seq))
            c = string(seq[j])
            idx = get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, c, 0)
            idx > 0 && (msa_one_hot[i, j, idx] = 1f0)
        end
    end

    # Deletion features
    has_deletion = zeros(Float32, n, L)
    deletion_value = zeros(Float32, n, L)
    if size(msa.deletion_matrix) == (n, L)
        for i in 1:n, j in 1:L
            d = msa.deletion_matrix[i, j]
            has_deletion[i, j] = Float32(d > 0)
            deletion_value[i, j] = Float32(atan(d / 3.0) * (2.0 / π))
        end
    end

    # MSA mask: non-gap positions
    msa_mask = zeros(Float32, n, L)
    for i in 1:n
        seq = msa.sequences[i]
        for j in 1:min(L, length(seq))
            msa_mask[i, j] = Float32(seq[j] != '-')
        end
    end

    # Profile and deletion mean
    profile, deletion_mean = compute_msa_profile(msa)

    return Dict{String,Array}(
        "msa_one_hot"     => msa_one_hot,
        "has_deletion"    => has_deletion,
        "deletion_value"  => deletion_value,
        "msa_mask"        => msa_mask,
        "msa_profile"     => profile,
        "deletion_mean"   => deletion_mean,
    )
end

"""
    make_extra_msa_features(msa::Msa; max_extra_msa=1024) -> Dict{String,Array}

Compute 25-channel features for the "extra" MSA (sequences beyond the main MSA).
"""
function make_extra_msa_features(msa::Msa; max_extra_msa::Int=1024)::Dict{String,Array}
    trunc_msa = truncate_msa(msa, max_extra_msa)
    n = n_seqs(trunc_msa)
    L = alignment_length(trunc_msa)

    # Compact representation: residue type as integer + deletion features
    msa_encoded = zeros(Int8, n, L)
    msa_mask    = ones(Bool, n, L)
    has_del     = zeros(Float32, n, L)
    del_val     = zeros(Float32, n, L)

    for i in 1:n
        seq = trunc_msa.sequences[i]
        for j in 1:min(L, length(seq))
            c = string(seq[j])
            idx = get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, c, 0)
            msa_encoded[i, j] = Int8(min(idx, 127))
            msa_mask[i, j]    = (c != "-")
        end
        if size(trunc_msa.deletion_matrix, 2) >= L
            for j in 1:L
                d = trunc_msa.deletion_matrix[i, j]
                has_del[i, j] = Float32(d > 0)
                del_val[i, j] = Float32(atan(d / 3.0) * (2.0 / π))
            end
        end
    end

    return Dict{String,Array}(
        "extra_msa"              => msa_encoded,
        "extra_msa_mask"         => msa_mask,
        "extra_has_deletion"     => has_del,
        "extra_deletion_value"   => del_val,
    )
end
