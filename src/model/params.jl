"""
params.jl — Model parameter loading from HDF5/NPZ files.
"""

using Logging
using HDF5

# ──────────────────────────────────────────────────────────────────────────────
# Expected parameter keys (structure matches Haiku-style parameter tree)
# ──────────────────────────────────────────────────────────────────────────────

const EXPECTED_PARAMS = Dict{String,Tuple{DataType,Vector{Int}}}(
    # Evoformer trunk — pair pairformer
    "evoformer/pairformer_stack/triangle_multiplication_outgoing/left_norm_input/scale" => (Float32, [128]),
    "evoformer/pairformer_stack/triangle_multiplication_outgoing/left_norm_input/offset" => (Float32, [128]),
    "evoformer/pairformer_stack/triangle_multiplication_outgoing/projection/weights"     => (Float32, [128, 256]),
    "evoformer/pairformer_stack/triangle_multiplication_outgoing/gate/weights"           => (Float32, [128, 256]),
    "evoformer/pairformer_stack/triangle_multiplication_outgoing/center_norm/scale"      => (Float32, [128]),
    "evoformer/pairformer_stack/triangle_multiplication_outgoing/output_projection/weights" => (Float32, [128, 128]),
    "evoformer/pairformer_stack/triangle_multiplication_outgoing/gating_linear/weights"  => (Float32, [128, 128]),
    "evoformer/pairformer_stack/triangle_multiplication_incoming/left_norm_input/scale"  => (Float32, [128]),
    "evoformer/pairformer_stack/grid_self_attention_row/query_norm/scale"                => (Float32, [128]),
    "evoformer/pairformer_stack/grid_self_attention_row/q_weights"                       => (Float32, [128, 4, 32]),
    "evoformer/pairformer_stack/grid_self_attention_row/k_weights"                       => (Float32, [128, 4, 32]),
    "evoformer/pairformer_stack/grid_self_attention_row/v_weights"                       => (Float32, [128, 4, 32]),
    "evoformer/pairformer_stack/grid_self_attention_row/gating_query/weights"            => (Float32, [128, 128]),
    "evoformer/pairformer_stack/grid_self_attention_row/output_projection/weights"       => (Float32, [128, 128]),
    "evoformer/pairformer_stack/pair_transition/transition_layer_norm/scale"             => (Float32, [128]),
    "evoformer/pairformer_stack/pair_transition/linear_1/weights"                        => (Float32, [128, 512]),
    "evoformer/pairformer_stack/pair_transition/linear_2/weights"                        => (Float32, [512, 128]),
    # Single activations
    "evoformer/pairformer_stack/single_attention/query_norm/scale"                       => (Float32, [384]),
    "evoformer/pairformer_stack/single_attention/q_weights"                              => (Float32, [384, 16, 24]),
    "evoformer/pairformer_stack/single_transition/transition_layer_norm/scale"           => (Float32, [384]),
    # Confidence head
    "confidence_head/left_target_feat_project/weights"                                   => (Float32, [384, 128]),
    "confidence_head/right_target_feat_project/weights"                                  => (Float32, [384, 128]),
    "confidence_head/plddt_logits/weights"                                               => (Float32, [384, 50]),
    "confidence_head/pae_logits/weights"                                                 => (Float32, [128, 64]),
    "confidence_head/pde_logits/weights"                                                 => (Float32, [128, 64]),
    # Diffusion head
    "diffusion_head/atom_transformer/self_attention/q_weights"                           => (Float32, [128, 16, 8]),
    "diffusion_head/position_update/weights"                                             => (Float32, [128, 3]),
)

# ──────────────────────────────────────────────────────────────────────────────
# bfloat16 conversion
# ──────────────────────────────────────────────────────────────────────────────

"""
    bfloat16_to_float32(arr::AbstractArray{UInt16}) -> Array{Float32}

Convert a UInt16 array storing bfloat16 values to Float32.
bfloat16 is stored as the upper 16 bits of Float32.
"""
function bfloat16_to_float32(arr::AbstractArray{UInt16})::Array{Float32}
    result = Array{Float32}(undef, size(arr))
    for i in eachindex(arr)
        # Shift bfloat16 bits to upper half of Float32
        u32 = UInt32(arr[i]) << 16
        result[i] = reinterpret(Float32, u32)
    end
    return result
end

"""
    get_param_f32(params::Dict, key::String) -> Array{Float32}

Get a parameter from the params dict, converting bfloat16 if necessary.
"""
function get_param_f32(params::Dict, key::String)::Array{Float32}
    val = get(params, key, nothing)
    val === nothing && error("Parameter not found: $key")
    val isa AbstractArray{UInt16} && return bfloat16_to_float32(val)
    return Float32.(val)
end

# ──────────────────────────────────────────────────────────────────────────────
# Parameter loading
# ──────────────────────────────────────────────────────────────────────────────

"""
    load_params(model_dir::String) -> Dict{String,Array}

Load model parameters from model_dir.
Looks for params.npz, params.h5, or sharded binary files.
"""
function load_params(model_dir::String)::Dict{String,Array}
    isdir(model_dir) || error("Model directory not found: $model_dir")

    # Try HDF5 first
    h5_path = joinpath(model_dir, "params.h5")
    isfile(h5_path) && return load_params_hdf5(h5_path)

    # Try serialized Julia dict
    bin_path = joinpath(model_dir, "params.bin")
    if isfile(bin_path)
        return safe_load(bin_path)
    end

    # Try NPZ-style files
    npz_path = joinpath(model_dir, "params.npz")
    isfile(npz_path) && return load_params_npz(npz_path)

    # Return empty params (for testing without checkpoint)
    @warn "No parameter files found in $model_dir; returning empty params"
    return Dict{String,Array}()
end

"""
    load_params_hdf5(path::String) -> Dict{String,Array}

Load parameters from an HDF5 file with flat-key format.
"""
function load_params_hdf5(path::String)::Dict{String,Array}
    params = Dict{String,Array}()
    h5open(path, "r") do f
        _read_hdf5_recursive!(params, f, "")
    end
    # Convert bfloat16 (stored as UInt16) to Float32
    for (k, v) in params
        v isa AbstractArray{UInt16} && (params[k] = bfloat16_to_float32(v))
    end
    return params
end

function _read_hdf5_recursive!(params::Dict, group, prefix::String)
    for name in keys(group)
        full_key = isempty(prefix) ? name : "$prefix/$name"
        item = group[name]
        if item isa HDF5.Dataset
            params[full_key] = read(item)
        elseif item isa HDF5.Group
            _read_hdf5_recursive!(params, item, full_key)
        end
    end
end

"""
    load_params_npz(path::String) -> Dict{String,Array}

Load parameters from a NumPy .npz archive.
"""
function load_params_npz(path::String)::Dict{String,Array}
    params = Dict{String,Array}()
    try
        # Use NPZ.jl if available
        @eval using NPZ
        data = NPZ.npzread(path)
        for (k, v) in data
            params[String(k)] = v
        end
    catch e
        @warn "Could not load NPZ file $path: $e"
    end
    return params
end

"""
    validate_params(params::Dict) -> Bool

Check that all expected parameter keys are present with correct shapes.
"""
function validate_params(params::Dict)::Bool
    all_ok = true
    for (key, (dtype, expected_shape)) in EXPECTED_PARAMS
        if !haskey(params, key)
            @debug "Missing parameter: $key (expected $(dtype)$(expected_shape))"
            all_ok = false
            continue
        end
        v = params[key]
        if collect(size(v)) != expected_shape
            @debug "Shape mismatch for $key: expected $expected_shape, got $(size(v))"
            all_ok = false
        end
    end
    return all_ok
end
