"""
vector.jl — Vec3Array struct-of-arrays for 3D vectors.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Vec3Array
# ──────────────────────────────────────────────────────────────────────────────

"""
    Vec3Array

Struct-of-arrays representation of an array of 3D vectors.
Each component (x, y, z) is an AbstractArray{Float32} of the same shape.
"""
struct Vec3Array{T<:AbstractArray{Float32}}
    x::T
    y::T
    z::T
end

function Vec3Array(x::AbstractArray, y::AbstractArray, z::AbstractArray)
    fx = Float32.(x)
    fy = Float32.(y)
    fz = Float32.(z)
    @assert size(fx) == size(fy) == size(fz) "Vec3Array: x/y/z must have the same shape"
    return Vec3Array{typeof(fx)}(fx, fy, fz)
end

function Base.show(io::IO, v::Vec3Array)
    print(io, "Vec3Array$(size(v.x))")
end

Base.size(v::Vec3Array) = size(v.x)
Base.eltype(::Vec3Array) = Float32

function Base.getindex(v::Vec3Array, idxs...)
    return Vec3Array(v.x[idxs...], v.y[idxs...], v.z[idxs...])
end

function Base.setindex!(v::Vec3Array, val::Vec3Array, idxs...)
    v.x[idxs...] = val.x
    v.y[idxs...] = val.y
    v.z[idxs...] = val.z
end

# ──────────────────────────────────────────────────────────────────────────────
# Arithmetic
# ──────────────────────────────────────────────────────────────────────────────

Base.:+(a::Vec3Array, b::Vec3Array) = Vec3Array(a.x .+ b.x, a.y .+ b.y, a.z .+ b.z)
Base.:-(a::Vec3Array, b::Vec3Array) = Vec3Array(a.x .- b.x, a.y .- b.y, a.z .- b.z)
Base.:-(v::Vec3Array)               = Vec3Array(-v.x, -v.y, -v.z)
Base.:*(s::Number, v::Vec3Array)    = Vec3Array(s .* v.x, s .* v.y, s .* v.z)
Base.:*(v::Vec3Array, s::Number)    = s * v
Base.:/(v::Vec3Array, s::Number)    = Vec3Array(v.x ./ s, v.y ./ s, v.z ./ s)

# ──────────────────────────────────────────────────────────────────────────────
# Geometric operations
# ──────────────────────────────────────────────────────────────────────────────

"""
    dot(a::Vec3Array, b::Vec3Array) -> AbstractArray{Float32}

Elementwise dot product.
"""
function dot(a::Vec3Array, b::Vec3Array)
    return a.x .* b.x .+ a.y .* b.y .+ a.z .* b.z
end

"""
    cross(a::Vec3Array, b::Vec3Array) -> Vec3Array

Elementwise cross product.
"""
function cross(a::Vec3Array, b::Vec3Array)
    return Vec3Array(
        a.y .* b.z .- a.z .* b.y,
        a.z .* b.x .- a.x .* b.z,
        a.x .* b.y .- a.y .* b.x,
    )
end

"""
    norm(v::Vec3Array; eps=1f-8) -> AbstractArray{Float32}

L2 norm of each vector.
"""
function norm(v::Vec3Array; eps::Float32=1f-8)
    return sqrt.(dot(v, v) .+ eps)
end

"""
    normalized(v::Vec3Array; eps=1f-10) -> Vec3Array

Unit vectors (L2-normalized).
"""
function normalized(v::Vec3Array; eps::Float32=1f-10)
    n = sqrt.(dot(v, v) .+ eps)
    return Vec3Array(v.x ./ n, v.y ./ n, v.z ./ n)
end

"""
    squared_norm(v::Vec3Array) -> AbstractArray{Float32}

Squared L2 norm.
"""
function squared_norm(v::Vec3Array)
    return dot(v, v)
end

# ──────────────────────────────────────────────────────────────────────────────
# Conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    from_array(arr::AbstractArray) -> Vec3Array

Construct a Vec3Array from an array whose last dimension has size 3.
"""
function vec3_from_array(arr::AbstractArray)::Vec3Array
    @assert size(arr)[end] == 3 "Last dimension must be 3, got $(size(arr))"
    idxs = ntuple(i -> Colon(), ndims(arr) - 1)
    return Vec3Array(
        Float32.(arr[idxs..., 1]),
        Float32.(arr[idxs..., 2]),
        Float32.(arr[idxs..., 3]),
    )
end

"""
    to_array(v::Vec3Array) -> AbstractArray

Convert Vec3Array to an array with last dimension = 3.
"""
function vec3_to_array(v::Vec3Array)::Array{Float32}
    return cat(
        reshape(v.x, size(v.x)..., 1),
        reshape(v.y, size(v.y)..., 1),
        reshape(v.z, size(v.z)..., 1);
        dims = ndims(v.x) + 1
    )
end

"""
    vec3_zeros(shape::Tuple; dtype=Float32) -> Vec3Array

Create a zero Vec3Array of the given shape.
"""
function vec3_zeros(shape::Tuple; dtype::Type=Float32)::Vec3Array
    return Vec3Array(
        zeros(dtype, shape),
        zeros(dtype, shape),
        zeros(dtype, shape),
    )
end

vec3_zeros(shape::Int...) = vec3_zeros(shape)

"""
    scalar_map(f, v::Vec3Array) -> Vec3Array

Apply scalar function to each component.
"""
function scalar_map(f, v::Vec3Array)::Vec3Array
    return Vec3Array(f.(v.x), f.(v.y), f.(v.z))
end

"""
    scalar_map2(f, a::Vec3Array, b::Vec3Array) -> Vec3Array

Apply binary scalar function to each component pair.
"""
function scalar_map2(f, a::Vec3Array, b::Vec3Array)::Vec3Array
    return Vec3Array(f.(a.x, b.x), f.(a.y, b.y), f.(a.z, b.z))
end
