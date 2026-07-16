"""
utils.jl — Miscellaneous geometry utilities.
"""

using LinearAlgebra

"""
    gram_schmidt_qr(v1::Vec3Array, v2::Vec3Array, v3::Vec3Array) -> Rot3Array

Compute the rotation matrix from three vectors using Gram-Schmidt + QR.
"""
function gram_schmidt_qr(v1::Vec3Array, v2::Vec3Array, v3::Vec3Array)::Rot3Array
    return rot_from_two_vectors(v1, v2)
end

"""
    make_backbone_frames(n_pos::Vec3Array, ca_pos::Vec3Array, c_pos::Vec3Array) -> Rigid3Array

Build backbone rigid frames from N, Cα, C positions.
Frame origin is at Cα; axes defined by Gram-Schmidt on (N-Cα, C-Cα).
"""
function make_backbone_frames(n_pos::Vec3Array, ca_pos::Vec3Array, c_pos::Vec3Array)::Rigid3Array
    e0 = n_pos - ca_pos   # N - Cα direction
    e1 = c_pos - ca_pos   # C - Cα direction
    rotation = rot_from_two_vectors(e0, e1)
    return Rigid3Array(rotation, ca_pos)
end

"""
    apply_rigid_to_atoms(rigid::Rigid3Array, atom_offsets::Vec3Array) -> Vec3Array

Apply rigid transforms to atom positions defined as offsets from the frame origin.
"""
function apply_rigid_to_atoms(rigid::Rigid3Array, atom_offsets::Vec3Array)::Vec3Array
    return rigid_apply_to_point(rigid, atom_offsets)
end

"""
    pairwise_distances(positions::AbstractMatrix{Float32}) -> AbstractMatrix{Float32}

Compute pairwise L2 distances between rows of a (N, 3) matrix.
"""
function pairwise_distances(positions::AbstractMatrix{Float32})::Matrix{Float32}
    n = size(positions, 1)
    dists = zeros(Float32, n, n)
    for i in 1:n, j in i+1:n
        d = sqrt(sum((positions[i,:] .- positions[j,:]).^2))
        dists[i,j] = d
        dists[j,i] = d
    end
    return dists
end

"""
    dihedral_angle(a, b, c, d) -> Float32

Compute the dihedral angle between planes (a,b,c) and (b,c,d).
All arguments are length-3 coordinate vectors.
"""
function dihedral_angle(a::AbstractVector, b::AbstractVector,
                        c::AbstractVector, d::AbstractVector)::Float32
    b1 = b .- a
    b2 = c .- b
    b3 = d .- c
    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    m1 = cross(n1, b2 ./ norm(b2))
    x = dot(n1, n2)
    y = dot(m1, n2)
    return Float32(atan(y, x))
end

"""
    unit_vector(v::Vec3Array) -> Vec3Array

Alias for normalized.
"""
unit_vector(v::Vec3Array) = normalized(v)

"""
    square(x) -> typeof(x)

Element-wise square.
"""
square(x) = x .* x

# Re-export norm from LinearAlgebra for plain vectors
import LinearAlgebra: norm, cross, dot
norm(v::AbstractVector) = LinearAlgebra.norm(v)
cross(a::AbstractVector, b::AbstractVector) = LinearAlgebra.cross(a, b)
dot(a::AbstractVector, b::AbstractVector) = LinearAlgebra.dot(a, b)
