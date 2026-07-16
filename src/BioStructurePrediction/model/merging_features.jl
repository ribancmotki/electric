"""
Feature merging utilities for combining per-chain features into batch tensors.
"""

"""
    pad_to_bucket(
        features::Dict{String,Array},
        bucket_size::Int
    ) -> Dict{String,Array}

Pad all token-indexed feature arrays to bucket_size along the token dimension.
Arrays are zero-padded and masks are set to 0 for padded positions.
"""
function pad_to_bucket(features::Dict{String,Array}, bucket_size::Int)::Dict{String,Array}
    padded = Dict{String,Array}()
    n_tokens = size(get(features, "token_index", zeros(Int32, 0)), 1)

    if n_tokens >= bucket_size
        # No padding needed (should not happen; bucket selection ensures n < bucket)
        return features
    end
    pad = bucket_size - n_tokens

    for (key, arr) in features
        if ndims(arr) == 0
            padded[key] = arr
        elseif size(arr, 1) == n_tokens
            # Pad along first dimension
            padding_shape = (pad, size(arr)[2:end]...)
            padded[key] = vcat(arr, zeros(eltype(arr), padding_shape...))
        elseif ndims(arr) >= 2 && size(arr, 2) == n_tokens && size(arr, 1) != n_tokens
            # MSA-shaped arrays: (n_seqs, n_tokens, ...)
            padding_shape = (size(arr, 1), pad, size(arr)[3:end]...)
            padded[key] = hcat(arr, zeros(eltype(arr), padding_shape...))
        else
            padded[key] = arr
        end
    end

    # Update seq_mask and token_index for padded positions
    if haskey(padded, "seq_mask")
        padded["seq_mask"] = vcat(features["seq_mask"], zeros(Float32, pad))
    end
    if haskey(padded, "token_index")
        last_idx = n_tokens > 0 ? features["token_index"][end] : 0
        padded["token_index"] = vcat(features["token_index"], Int32.(last_idx+1:last_idx+pad))
    end

    return padded
end

"""
    select_bucket(n_tokens::Int, buckets::Vector{Int}) -> Int

Select the smallest bucket that can accommodate n_tokens.
Raises an error if n_tokens exceeds all buckets.
"""
function select_bucket(n_tokens::Int, buckets::Vector{Int})::Int
    for b in sort(buckets)
        b >= n_tokens && return b
    end
    error("Token count $n_tokens exceeds all bucket sizes $(maximum(buckets)). " *
          "Use a larger bucket or reduce input size.")
end

"""
    merge_batch_dicts(dicts::Vector{Dict{String,Array}}) -> Dict{String,Array}

Stack a list of feature dicts (one per seed) into a single batched dict.
Arrays are stacked along a new leading batch dimension.
"""
function merge_batch_dicts(dicts::Vector{Dict{String,Array}})::Dict{String,Array}
    isempty(dicts) && return Dict{String,Array}()
    length(dicts) == 1 && return dicts[1]

    merged = Dict{String,Array}()
    for key in keys(first(dicts))
        arrays = [d[key] for d in dicts if haskey(d, key)]
        isempty(arrays) && continue
        if ndims(first(arrays)) == 0
            merged[key] = first(arrays)
        else
            merged[key] = cat(arrays...; dims=ndims(first(arrays))+1)
        end
    end
    return merged
end

"""
    split_batch_dict(batch::Dict{String,Array}) -> Vector{Dict{String,Array}}

Split a batched feature dict (last dim = batch) into per-sample dicts.
"""
function split_batch_dict(batch::Dict{String,Array})::Vector{Dict{String,Array}}
    # Determine batch size
    n_batch = 1
    for (_, arr) in batch
        if ndims(arr) > 0
            n_batch = size(arr, ndims(arr))
            break
        end
    end

    result = [Dict{String,Array}() for _ in 1:n_batch]
    for (key, arr) in batch
        if ndims(arr) == 0
            for d in result
                d[key] = arr
            end
        else
            for b in 1:n_batch
                idx = ntuple(_ -> Colon(), ndims(arr)-1)
                result[b][key] = arr[idx..., b]
            end
        end
    end
    return result
end
