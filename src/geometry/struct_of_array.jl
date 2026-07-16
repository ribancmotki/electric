"""
struct_of_array.jl — Struct-of-arrays utilities for geometry types.
"""

"""
    soa_map(f, soa)

Apply function `f` to each scalar field of a struct-of-arrays, returning a new struct.
Used internally by Vec3Array, Rot3Array, Rigid3Array for elementwise operations.
"""
macro soa_map(T, expr_pairs...)
    # Helper macro — not exported, used internally
end

"""
    broadcast_shapes(a, b) -> Tuple

Compute the broadcast shape of two shapes.
"""
function broadcast_shapes(a::Tuple, b::Tuple)::Tuple
    la, lb = length(a), length(b)
    n = max(la, lb)
    a = (ones(Int, n-la)..., a...)
    b = (ones(Int, n-lb)..., b...)
    result = ntuple(n) do i
        ai, bi = a[i], b[i]
        (ai == 1) && return bi
        (bi == 1) && return ai
        @assert ai == bi "Incompatible shapes: $a vs $b"
        ai
    end
    return result
end

"""
    batch_shape(a::AbstractArray, n_batch_dims::Int) -> Tuple

Extract the batch dimensions from an array.
"""
function batch_shape(a::AbstractArray, n_batch_dims::Int)::Tuple
    return size(a)[1:n_batch_dims]
end
