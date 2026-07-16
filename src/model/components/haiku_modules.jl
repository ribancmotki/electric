"""
haiku_modules.jl — Custom neural network layers (Linear, LayerNorm).
"""

using Flux
using LinearAlgebra

# Truncated normal correction factor (from JAX/Haiku)
const TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978f0

# ──────────────────────────────────────────────────────────────────────────────
# Linear layer
# ──────────────────────────────────────────────────────────────────────────────

"""
    Linear

Dense linear layer with configurable initialization and weight transpose.
"""
struct Linear
    weight::AbstractArray{Float32}
    bias::Union{AbstractVector{Float32},Nothing}
    num_output::Union{Int,Tuple}
    num_input_dims::Int
    transpose_weights::Bool
end

function Linear(num_output::Union{Int,Tuple};
    num_input_dims::Int = 1,
    use_bias::Bool      = false,
    bias_init::Float32  = 0f0,
    initializer::Symbol = :linear,
    precision::Symbol   = :default,
    transpose_weights::Bool = false,
    in_channels::Union{Int,Nothing} = nothing,
)
    # Placeholder: weights initialized lazily based on input shape at first call.
    # For actual use, the weight matrix is created when we know the input size.
    return Linear(Float32[], Float32[], num_output, num_input_dims, transpose_weights)
end

function (layer::Linear)(x::AbstractArray{Float32})::AbstractArray{Float32}
    # x shape: (..., in_features)
    sh = size(x)
    in_dim = sh[end]

    # Lazy weight initialization using He/fan_in scaling
    w = _get_or_create_weights(layer, in_dim)
    b = layer.bias

    # Matrix multiply: reshape x to (batch, in_dim), then mul with w
    batch_sh = sh[1:end-1]
    x_2d = reshape(x, :, in_dim)

    if layer.transpose_weights
        out_2d = x_2d * w'  # (batch, out_dim)
    else
        out_2d = x_2d * w   # (batch, out_dim)
    end

    if b !== nothing && length(b) > 0
        out_2d = out_2d .+ b'
    end

    out_dim = size(out_2d, 2)
    if layer.num_output isa Tuple
        return reshape(out_2d, batch_sh..., layer.num_output...)
    else
        return reshape(out_2d, batch_sh..., out_dim)
    end
end

function _get_or_create_weights(layer::Linear, in_dim::Int)::Matrix{Float32}
    out_dim = layer.num_output isa Tuple ? prod(layer.num_output) : layer.num_output
    # Default: create new random weights (in production, weights are loaded from params)
    return randn(Float32, in_dim, out_dim) .* (1f0 / sqrt(Float32(in_dim))) .* TRUNCATED_NORMAL_STDDEV_FACTOR
end

# ──────────────────────────────────────────────────────────────────────────────
# Parameterized Linear (used in model)
# ──────────────────────────────────────────────────────────────────────────────

"""
    DenseLinear

A concrete linear layer with stored parameters (used in Flux models).
"""
mutable struct DenseLinear
    W::Matrix{Float32}   # (in_dim, out_dim)
    b::Union{Vector{Float32},Nothing}
end

Flux.@functor DenseLinear

function DenseLinear(in_dim::Int, out_dim::Int;
    use_bias::Bool  = false,
    initializer::Symbol = :linear,
    bias_init::Float32 = 0f0,
)
    stddev = if initializer == :linear
        TRUNCATED_NORMAL_STDDEV_FACTOR / sqrt(Float32(in_dim))
    elseif initializer == :relu
        TRUNCATED_NORMAL_STDDEV_FACTOR * sqrt(2f0 / Float32(in_dim))
    else  # :zeros
        0f0
    end
    W = stddev > 0 ? randn(Float32, in_dim, out_dim) .* stddev : zeros(Float32, in_dim, out_dim)
    b = use_bias ? fill(bias_init, out_dim) : nothing
    return DenseLinear(W, b)
end

function (l::DenseLinear)(x::AbstractArray{Float32})
    sh = size(x)
    in_dim = sh[end]
    @assert in_dim == size(l.W, 1) "Input dim $in_dim != weight dim $(size(l.W,1))"
    x_2d = reshape(x, :, in_dim)
    out_2d = x_2d * l.W
    l.b !== nothing && (out_2d = out_2d .+ l.b')
    return reshape(out_2d, sh[1:end-1]..., size(l.W, 2))
end

function Base.show(io::IO, l::DenseLinear)
    print(io, "DenseLinear($(size(l.W,1)) → $(size(l.W,2)))")
end

# ──────────────────────────────────────────────────────────────────────────────
# LayerNorm
# ──────────────────────────────────────────────────────────────────────────────

"""
    StructLayerNorm

Layer normalization with configurable scale/offset and bfloat16 upcasting.
"""
mutable struct StructLayerNorm
    scale::Union{Vector{Float32},Nothing}
    offset::Union{Vector{Float32},Nothing}
    axis::Int
    eps::Float32
    upcast::Bool
    create_scale::Bool
    create_offset::Bool
end

Flux.@functor StructLayerNorm (scale, offset)

function StructLayerNorm(n_features::Int;
    axis::Int    = -1,
    create_scale::Bool  = true,
    create_offset::Bool = true,
    eps::Float32        = 1f-5,
    upcast::Bool        = true,
)
    scale  = create_scale  ? ones(Float32, n_features) : nothing
    offset = create_offset ? zeros(Float32, n_features) : nothing
    return StructLayerNorm(scale, offset, axis, eps, upcast, create_scale, create_offset)
end

function (ln::StructLayerNorm)(x::AbstractArray{T}) where T
    # Upcast to Float32 if needed
    xf = ln.upcast ? Float32.(x) : x

    # Normalize along last axis (axis=-1)
    μ = mean(xf, dims=ndims(xf))
    σ2 = mean((xf .- μ).^2, dims=ndims(xf))
    xn = (xf .- μ) ./ sqrt.(σ2 .+ ln.eps)

    # Apply learnable scale and offset
    if ln.scale !== nothing
        xn = xn .* reshape(ln.scale, ntuple(_->1, ndims(x)-1)..., :)
    end
    if ln.offset !== nothing
        xn = xn .+ reshape(ln.offset, ntuple(_->1, ndims(x)-1)..., :)
    end

    return T === Float32 ? xn : T.(xn)
end

function Base.show(io::IO, ln::StructLayerNorm)
    n = ln.scale !== nothing ? length(ln.scale) : 0
    print(io, "StructLayerNorm($n, scale=$(ln.create_scale), offset=$(ln.create_offset))")
end

# ──────────────────────────────────────────────────────────────────────────────
# Activation functions
# ──────────────────────────────────────────────────────────────────────────────

"""swish activation: x * sigmoid(x)"""
swish(x) = x .* sigmoid.(x)

"""gated linear unit: swish(a) * b where [a, b] = split(x, 2)"""
function glu(x::AbstractArray)
    n = size(x, ndims(x))
    @assert iseven(n) "GLU input last dim must be even"
    h = n ÷ 2
    a = selectdim(x, ndims(x), 1:h)
    b = selectdim(x, ndims(x), h+1:n)
    return swish(a) .* b
end
