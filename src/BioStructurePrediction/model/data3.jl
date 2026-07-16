"""
Supplementary data structures and utilities for model I/O.
"""

"""
    ModelResult

Type alias for the raw output dictionary from the model forward pass.
"""
const ModelResult = Dict{String,Any}

"""
    make_empty_model_result(n_tokens::Int, n_samples::Int) -> ModelResult

Construct an empty ModelResult with correctly-shaped zero arrays.
Used for testing and as a fallback when inference is not available.
"""
function make_empty_model_result(n_tokens::Int, n_samples::Int)::ModelResult
    return ModelResult(
        "predicted_positions"          => zeros(Float32, n_samples, n_tokens, NUM_ATOM_SLOTS, 3),
        "predicted_atom_mask"          => zeros(Float32, n_samples, n_tokens, NUM_ATOM_SLOTS),
        "plddt_logits"                 => zeros(Float32, n_tokens, NUM_ATOM_TYPES_PLDDT, PLDDT_NUM_BINS),
        "pae_logits"                   => zeros(Float32, n_tokens, n_tokens, PAE_NUM_BINS),
        "experimentally_resolved_logits" => zeros(Float32, n_tokens, NUM_ATOM_TYPES_PLDDT, 2),
        "distogram"                    => Dict{String,Any}("distogram" => zeros(Float32, n_tokens, n_tokens, DISTOGRAM_NUM_BINS)),
        "single_embeddings"            => zeros(Float32, n_tokens, SINGLE_FEAT_DIM),
        "pair_embeddings"              => zeros(Float32, n_tokens, n_tokens, PAIR_FEAT_DIM),
        "__identifier__"               => UInt8[],
    )
end

"""
    remove_invalidly_typed_feats(batch::BatchDict) -> BatchDict

Remove feature arrays that have element types incompatible with GPU execution.
Returns a filtered copy of the batch dict.
"""
function remove_invalidly_typed_feats(batch::BatchDict)::BatchDict
    valid_types = Union{Float32, Float16, Int32, Int64, Bool, UInt64}
    filtered = BatchDict()
    for (key, arr) in batch
        if arr isa AbstractArray
            if eltype(arr) <: Number
                filtered[key] = arr
            else
                # String arrays and other non-numeric arrays are excluded
                @debug "Removing non-numeric feature '$key' ($(eltype(arr)))"
            end
        else
            # Scalars
            filtered[key] = arr
        end
    end
    return filtered
end

"""
    move_batch_to_gpu(batch::BatchDict) -> BatchDict

Move all numeric arrays in the batch to GPU memory.
"""
function move_batch_to_gpu(batch::BatchDict)::BatchDict
    using CUDA
    gpu_batch = BatchDict()
    for (key, arr) in batch
        if arr isa AbstractArray && eltype(arr) <: Number
            try
                gpu_batch[key] = CUDA.cu(arr)
            catch e
                @warn "Could not move '$key' to GPU: $e"
                gpu_batch[key] = arr
            end
        else
            gpu_batch[key] = arr
        end
    end
    return gpu_batch
end

"""
    move_result_to_cpu(result::ModelResult) -> ModelResult

Move all arrays in the model result from GPU to CPU.
"""
function move_result_to_cpu(result::ModelResult)::ModelResult
    cpu_result = ModelResult()
    for (key, val) in result
        if val isa AbstractArray
            try
                cpu_result[key] = Array(val)
            catch
                cpu_result[key] = val
            end
        elseif val isa Dict
            cpu_result[key] = move_result_to_cpu(val)
        else
            cpu_result[key] = val
        end
    end
    return cpu_result
end

"""
    convert_bfloat16_to_float32(result::ModelResult) -> ModelResult

Convert any UInt16-typed arrays (bfloat16) in the result to Float32.
"""
function convert_bfloat16_to_float32(result::ModelResult)::ModelResult
    converted = ModelResult()
    for (key, val) in result
        if val isa AbstractArray && eltype(val) == UInt16
            converted[key] = bfloat16_to_float32(Array(val))
        elseif val isa Dict
            converted[key] = convert_bfloat16_to_float32(val)
        else
            converted[key] = val
        end
    end
    return converted
end
