"""
Rigid body geometry operations for the diffusion head.
Implemented as pure Julia analogues of JAX geometry utilities.
"""

using LinearAlgebra

# ──────────────────────────────────────────────
#  Rigid3Array
# ──────────────────────────────────────────────

"""
    Rigid3Array

A rigid body transformation: rotation matrix (3×3) + translation (3,).
All operations use Float32.
"""
struct Rigid3Array
    rotation::Matrix{Float32}    # 3×3 orthonormal rotation matrix
    translation::Vector{Float32} # length 3
end

"""
    identity_rigid() -> Rigid3Array

Return the identity rigid body transformation.
"""
function identity_rigid()::Rigid3Array
    return Rigid3Array(Matrix{Float32}(I, 3, 3), zeros(Float32, 3))
end

"""
    compose(r1::Rigid3Array, r2::Rigid3Array) -> Rigid3Array

Compose two rigid transformations: apply r2 first, then r1.
compose(r1, r2)(x) = r1(r2(x)) = r1.R * r2.R * x + r1.R * r2.t + r1.t
"""
function compose(r1::Rigid3Array, r2::Rigid3Array)::Rigid3Array
    R = r1.rotation * r2.rotation
    t = r1.rotation * r2.translation .+ r1.translation
    return Rigid3Array(R, t)
end

"""
    apply(r::Rigid3Array, points::AbstractMatrix{Float32}) -> Matrix{Float32}

Apply rigid transformation to a set of points.
points: (n_points, 3)
Returns: (n_points, 3)
"""
function apply(r::Rigid3Array, points::AbstractMatrix{Float32})::Matrix{Float32}
    # points: (n, 3) — each row is a point
    return (r.rotation * points')' .+ reshape(r.translation, 1, 3)
end

"""
    apply(r::Rigid3Array, point::AbstractVector{Float32}) -> Vector{Float32}

Apply rigid transformation to a single point.
"""
function apply(r::Rigid3Array, point::AbstractVector{Float32})::Vector{Float32}
    return r.rotation * point .+ r.translation
end

"""
    invert(r::Rigid3Array) -> Rigid3Array

Return the inverse transformation: R^T, -R^T * t.
"""
function invert(r::Rigid3Array)::Rigid3Array
    R_inv = r.rotation'
    t_inv = -(R_inv * r.translation)
    return Rigid3Array(R_inv, t_inv)
end

# ──────────────────────────────────────────────
#  Gram-Schmidt orthonormalisation
# ──────────────────────────────────────────────

"""
    gram_schmidt_qr(a::Vector{Float32}, b::Float32) -> Matrix{Float32}

Construct a rotation matrix from two (possibly non-orthogonal) vectors a and b,
using the Gram-Schmidt / modified-QR process.

Returns a 3×3 rotation matrix whose columns form an orthonormal basis:
  e1 = normalise(a)
  e2 = normalise(b - (b·e1) * e1)
  e3 = cross(e1, e2)
"""
function gram_schmidt_qr(a::Vector{Float32}, b::Vector{Float32})::Matrix{Float32}
    eps = 1f-8

    # e1
    a_norm = norm(a)
    e1 = a_norm > eps ? a ./ a_norm : Float32[1, 0, 0]

    # e2: Gram-Schmidt
    b_proj = b .- dot(b, e1) .* e1
    b_norm = norm(b_proj)
    e2 = b_norm > eps ? b_proj ./ b_norm : Float32[0, 1, 0]

    # e3: cross product
    e3 = Float32[
        e1[2]*e2[3] - e1[3]*e2[2],
        e1[3]*e2[1] - e1[1]*e2[3],
        e1[1]*e2[2] - e1[2]*e2[1],
    ]

    return hcat(e1, e2, e3)  # 3×3 matrix, columns are basis vectors
end

# ──────────────────────────────────────────────
#  Backbone frames
# ──────────────────────────────────────────────

"""
    make_backbone_frames(
        n_pos::AbstractMatrix{Float32},
        ca_pos::AbstractMatrix{Float32},
        c_pos::AbstractMatrix{Float32}
    ) -> Vector{Rigid3Array}

Construct local backbone frames from N, Cα, C atom positions.

n_pos, ca_pos, c_pos: each (n_residues, 3)
Returns: vector of n_residues Rigid3Array objects.

Frame definition:
- Origin at Cα
- e1 along Cα → C (backbone direction)
- e2 in the plane of N-Cα-C, perpendicular to e1
- e3 = cross(e1, e2)
"""
function make_backbone_frames(
    n_pos::AbstractMatrix{Float32},
    ca_pos::AbstractMatrix{Float32},
    c_pos::AbstractMatrix{Float32},
)::Vector{Rigid3Array}
    n_res = size(ca_pos, 1)
    size(n_pos, 1)  == n_res || error("n_pos has $(size(n_pos, 1)) residues, expected $n_res")
    size(c_pos, 1)  == n_res || error("c_pos has $(size(c_pos, 1)) residues, expected $n_res")

    frames = Vector{Rigid3Array}(undef, n_res)
    for i in 1:n_res
        ca = Float32.(ca_pos[i, :])
        c  = Float32.(c_pos[i, :])
        n  = Float32.(n_pos[i, :])

        # e1: Cα → C
        a = c .- ca
        # e2 candidate: Cα → N, projected orthogonal to e1
        b = n .- ca

        R = gram_schmidt_qr(a, b)
        frames[i] = Rigid3Array(R, ca)
    end
    return frames
end

"""
    apply_rigid_to_atoms(
        frames::Vector{Rigid3Array},
        local_positions::AbstractArray{Float32,3}
    ) -> Array{Float32,3}

Apply per-residue frames to local atom positions.

frames: length n_residues
local_positions: (n_residues, n_atoms, 3) in local frame
Returns: (n_residues, n_atoms, 3) in global frame
"""
function apply_rigid_to_atoms(
    frames::Vector{Rigid3Array},
    local_positions::AbstractArray{Float32,3},
)::Array{Float32,3}
    n_res, n_atoms, _ = size(local_positions)
    global_positions = zeros(Float32, n_res, n_atoms, 3)
    for i in 1:n_res
        for j in 1:n_atoms
            pt = Float32.(local_positions[i, j, :])
            global_positions[i, j, :] = apply(frames[i], pt)
        end
    end
    return global_positions
end

# ──────────────────────────────────────────────
#  Distance and angle utilities
# ──────────────────────────────────────────────

"""
    pairwise_distances(pos::AbstractMatrix{Float32}) -> Matrix{Float32}

Compute pairwise Euclidean distance matrix.
pos: (n, 3) — returns (n, n) symmetric distance matrix.
"""
function pairwise_distances(pos::AbstractMatrix{Float32})::Matrix{Float32}
    n = size(pos, 1)
    D = zeros(Float32, n, n)
    for i in 1:n, j in i+1:n
        d = sqrt(sum((pos[i, :] .- pos[j, :]).^2))
        D[i, j] = d
        D[j, i] = d
    end
    return D
end

"""
    unit_vector(v::Vector{Float32}) -> Vector{Float32}

Return the unit vector of v, or [1,0,0] if |v| < eps.
"""
function unit_vector(v::Vector{Float32})::Vector{Float32}
    n = norm(v)
    return n < 1f-8 ? Float32[1, 0, 0] : v ./ n
end

"""
    dihedral_angle(
        a::Vector{Float32}, b::Vector{Float32},
        c::Vector{Float32}, d::Vector{Float32}
    ) -> Float32

Compute the dihedral angle (in radians) defined by atoms a-b-c-d.
"""
function dihedral_angle(
    a::Vector{Float32}, b::Vector{Float32},
    c::Vector{Float32}, d::Vector{Float32}
)::Float32
    b1 = b .- a
    b2 = c .- b
    b3 = d .- c

    n1 = Float32[b1[2]*b2[3]-b1[3]*b2[2], b1[3]*b2[1]-b1[1]*b2[3], b1[1]*b2[2]-b1[2]*b2[1]]
    n2 = Float32[b2[2]*b3[3]-b2[3]*b3[2], b2[3]*b3[1]-b2[1]*b3[3], b2[1]*b3[2]-b2[2]*b3[1]]

    m1 = Float32[n1[2]*b2[3]-n1[3]*b2[2], n1[3]*b2[1]-n1[1]*b2[3], n1[1]*b2[2]-n1[2]*b2[1]]

    x = dot(n1, n2)
    y = dot(m1, n2)
    return atan(y, x)
end
