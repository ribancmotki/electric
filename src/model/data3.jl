"""
data3.jl — ModelResult helpers, GPU transfer, and validation.
"""

using Logging

# ──────────────────────────────────────────────────────────────────────────────
# ModelResult (Dict{String,Any}) helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    get_result_field(result::ModelResult, key::String, default=nothing)

Safe accessor for model result fields.
"""
function get_result_field(result::ModelResult, key::String, default=nothing)
    return get(result, key, default)
end

"""
    merge_model_results(results::Vector{ModelResult}) -> ModelResult

Merge list of model results by appending per-sample keys.
"""
function merge_model_results(results::Vector{ModelResult})::ModelResult
    isempty(results) && return ModelResult()
    length(results) == 1 && return results[1]

    merged = ModelResult()
    for key in keys(results[1])
        values_list = [r[key] for r in results if haskey(r, key)]
        isempty(values_list) && continue
        if all(v isa AbstractArray for v in values_list)
            try
                merged[key] = stack_arrays(values_list)
                continue
            catch; end
        end
        merged[key] = values_list
    end
    return merged
end

# ──────────────────────────────────────────────────────────────────────────────
# GPU transfer (no-op on CPU)
# ──────────────────────────────────────────────────────────────────────────────

"""
    move_batch_to_gpu(batch::Dict) -> Dict

Move all array values in `batch` to the current CUDA device.
On CPU builds, returns batch unchanged.
"""
function move_batch_to_gpu(batch::Dict)::Dict
    # Attempt CUDA.jl transfer if available
    cuda_available = false
    try
        @eval using CUDA
        cuda_available = CUDA.functional()
    catch
        cuda_available = false
    end

    cuda_available || return batch

    result = Dict{String,Any}()
    for (k, v) in batch
        if v isa AbstractArray
            try
                result[k] = CUDA.CuArray(v)
            catch
                result[k] = v
            end
        else
            result[k] = v
        end
    end
    return result
end

"""
    move_result_to_cpu(result::ModelResult) -> ModelResult

Move all CUDA arrays in `result` back to CPU.
"""
function move_result_to_cpu(result::ModelResult)::ModelResult
    cpu_result = ModelResult()
    for (k, v) in result
        if v isa AbstractArray
            try
                cpu_result[k] = Array(v)
            catch
                cpu_result[k] = v
            end
        elseif v isa Dict
            cpu_result[k] = move_result_to_cpu(v)
        else
            cpu_result[k] = v
        end
    end
    return cpu_result
end

# ──────────────────────────────────────────────────────────────────────────────
# Batch validation
# ──────────────────────────────────────────────────────────────────────────────

"""
    validate_batch(batch::Dict) -> Vector{String}

Return list of validation warnings for a batch dict.
"""
function validate_batch(batch::Dict)::Vector{String}
    warnings = String[]

    n_tokens = get(batch, "num_tokens", nothing)
    n_tokens === nothing && push!(warnings, "num_tokens not set")

    tf = get(batch, "token_features", nothing)
    if tf === nothing
        push!(warnings, "token_features not present")
    else
        for key in ["aatype", "residue_index", "asym_id"]
            if !haskey(tf, key)
                push!(warnings, "token_features missing '$key'")
            end
        end
    end

    # Check reference structure
    for key in ["ref_pos", "ref_mask"]
        if !haskey(batch, key)
            push!(warnings, "Missing reference structure field '$key'")
        end
    end

    return warnings
end

"""
    add_recycling_dims(batch::Dict) -> Dict

Add a recycle dimension (size 1) to all token-level arrays, matching
expected model input shapes.
"""
function add_recycling_dims(batch::Dict)::Dict
    result = Dict{String,Any}()
    n_tokens = get(batch, "num_tokens", 0)
    for (k, v) in batch
        if v isa AbstractArray && ndims(v) > 0 && size(v, 1) == n_tokens
            result[k] = reshape(v, 1, size(v)...)
        else
            result[k] = v
        end
    end
    return result
end
