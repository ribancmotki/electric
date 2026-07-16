"""
rotation_matrix.jl — Rot3Array struct-of-arrays for 3D rotation matrices.
"""

using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# Rot3Array
# ──────────────────────────────────────────────────────────────────────────────

"""
    Rot3Array

Struct-of-arrays representation of an array of 3×3 rotation matrices.
Each of the 9 scalar fields (xx..zz) is an AbstractArray{Float32} of the same batch shape.
"""
struct Rot3Array{T<:AbstractArray{Float32}}
    xx::T; xy::T; xz::T
    yx::T; yy::T; yz::T
    zx::T; zy::T; zz::T
end

function Rot3Array(xx, xy, xz, yx, yy, yz, zx, zy, zz)
    xs = map(x -> Float32.(x), (xx, xy, xz, yx, yy, yz, zx, zy, zz))
    @assert length(Set(size.(xs))) == 1 "All components must have the same shape"
    return Rot3Array{typeof(xs[1])}(xs...)
end

function Base.show(io::IO, r::Rot3Array)
    print(io, "Rot3Array$(size(r.xx))")
end

Base.size(r::Rot3Array) = size(r.xx)

function Base.getindex(r::Rot3Array, idxs...)
    return Rot3Array(
        r.xx[idxs...], r.xy[idxs...], r.xz[idxs...],
        r.yx[idxs...], r.yy[idxs...], r.yz[idxs...],
        r.zx[idxs...], r.zy[idxs...], r.zz[idxs...],
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Operations
# ──────────────────────────────────────────────────────────────────────────────

"""
    inverse(r::Rot3Array) -> Rot3Array

Compute the inverse (transpose) of a rotation matrix.
"""
function rot_inverse(r::Rot3Array)::Rot3Array
    return Rot3Array(
        r.xx, r.yx, r.zx,
        r.xy, r.yy, r.zy,
        r.xz, r.yz, r.zz,
    )
end

"""
    apply_to_point(r::Rot3Array, p::Vec3Array) -> Vec3Array

Apply rotation to a point (matrix-vector product).
"""
function rot_apply_to_point(r::Rot3Array, p::Vec3Array)::Vec3Array
    return Vec3Array(
        r.xx .* p.x .+ r.xy .* p.y .+ r.xz .* p.z,
        r.yx .* p.x .+ r.yy .* p.y .+ r.yz .* p.z,
        r.zx .* p.x .+ r.zy .* p.y .+ r.zz .* p.z,
    )
end

"""
    apply_inverse_to_point(r::Rot3Array, p::Vec3Array) -> Vec3Array

Apply the inverse rotation to a point.
"""
function rot_apply_inverse_to_point(r::Rot3Array, p::Vec3Array)::Vec3Array
    return rot_apply_to_point(rot_inverse(r), p)
end

"""
    rot_compose(a::Rot3Array, b::Rot3Array) -> Rot3Array

Compose two rotations: R = A * B.
"""
function rot_compose(a::Rot3Array, b::Rot3Array)::Rot3Array
    return Rot3Array(
        a.xx .* b.xx .+ a.xy .* b.yx .+ a.xz .* b.zx,
        a.xx .* b.xy .+ a.xy .* b.yy .+ a.xz .* b.zy,
        a.xx .* b.xz .+ a.xy .* b.yz .+ a.xz .* b.zz,
        a.yx .* b.xx .+ a.yy .* b.yx .+ a.yz .* b.zx,
        a.yx .* b.xy .+ a.yy .* b.yy .+ a.yz .* b.zy,
        a.yx .* b.xz .+ a.yy .* b.yz .+ a.yz .* b.zz,
        a.zx .* b.xx .+ a.zy .* b.yx .+ a.zz .* b.zx,
        a.zx .* b.xy .+ a.zy .* b.yy .+ a.zz .* b.zy,
        a.zx .* b.xz .+ a.zy .* b.yz .+ a.zz .* b.zz,
    )
end

Base.:*(a::Rot3Array, b::Rot3Array) = rot_compose(a, b)

# ──────────────────────────────────────────────────────────────────────────────
# Constructors
# ──────────────────────────────────────────────────────────────────────────────

"""
    rot_identity(shape::Tuple; dtype=Float32) -> Rot3Array

Identity rotation for a given batch shape.
"""
function rot_identity(shape::Tuple; dtype::Type=Float32)::Rot3Array
    o = zeros(dtype, shape)
    i = ones(dtype, shape)
    return Rot3Array(i, o, o, o, i, o, o, o, i)
end

rot_identity(shape::Int...) = rot_identity(shape)

"""
    rot_from_two_vectors(e0::Vec3Array, e1::Vec3Array) -> Rot3Array

Gram-Schmidt orthonormalization of two vectors to produce a rotation matrix.
e0 is the first axis; e1 is orthogonalized against e0, then e2 = e0 × e1.
"""
function rot_from_two_vectors(e0::Vec3Array, e1::Vec3Array)::Rot3Array
    # Normalize e0
    e0n = normalized(e0)
    # Orthogonalize e1 against e0
    dot_e1_e0 = dot(e1, e0n)
    e1_orth = Vec3Array(
        e1.x .- dot_e1_e0 .* e0n.x,
        e1.y .- dot_e1_e0 .* e0n.y,
        e1.z .- dot_e1_e0 .* e0n.z,
    )
    e1n = normalized(e1_orth)
    # e2 = e0 × e1
    e2n = cross(e0n, e1n)
    return Rot3Array(
        e0n.x, e1n.x, e2n.x,
        e0n.y, e1n.y, e2n.y,
        e0n.z, e1n.z, e2n.z,
    )
end

"""
    rot_from_array(arr::AbstractArray) -> Rot3Array

Construct from array of shape (..., 3, 3).
"""
function rot_from_array(arr::AbstractArray)::Rot3Array
    @assert size(arr)[end-1:end] == (3,3) "Last two dims must be (3,3)"
    batch = size(arr)[1:end-2]
    idxs  = ntuple(i -> Colon(), length(batch))
    return Rot3Array(
        Float32.(arr[idxs..., 1, 1]), Float32.(arr[idxs..., 1, 2]), Float32.(arr[idxs..., 1, 3]),
        Float32.(arr[idxs..., 2, 1]), Float32.(arr[idxs..., 2, 2]), Float32.(arr[idxs..., 2, 3]),
        Float32.(arr[idxs..., 3, 1]), Float32.(arr[idxs..., 3, 2]), Float32.(arr[idxs..., 3, 3]),
    )
end

"""
    rot_to_array(r::Rot3Array) -> Array{Float32}

Convert to array of shape (..., 3, 3).
"""
function rot_to_array(r::Rot3Array)::Array{Float32}
    sh = size(r.xx)
    result = zeros(Float32, sh..., 3, 3)
    result[ntuple(i->Colon(),length(sh))..., 1, 1] = r.xx
    result[ntuple(i->Colon(),length(sh))..., 1, 2] = r.xy
    result[ntuple(i->Colon(),length(sh))..., 1, 3] = r.xz
    result[ntuple(i->Colon(),length(sh))..., 2, 1] = r.yx
    result[ntuple(i->Colon(),length(sh))..., 2, 2] = r.yy
    result[ntuple(i->Colon(),length(sh))..., 2, 3] = r.yz
    result[ntuple(i->Colon(),length(sh))..., 3, 1] = r.zx
    result[ntuple(i->Colon(),length(sh))..., 3, 2] = r.zy
    result[ntuple(i->Colon(),length(sh))..., 3, 3] = r.zz
    return result
end

"""
    rot_from_quaternion(w, x, y, z; normalize=true, epsilon=1f-6) -> Rot3Array

Convert quaternion (w, x, y, z) to rotation matrix.
"""
function rot_from_quaternion(w, x, y, z;
                             normalize::Bool=true, epsilon::Float32=1f-6)::Rot3Array
    if normalize
        n = sqrt.(w.^2 .+ x.^2 .+ y.^2 .+ z.^2 .+ epsilon)
        w, x, y, z = w./n, x./n, y./n, z./n
    end
    return Rot3Array(
        1 .- 2 .* (y.^2 .+ z.^2),
        2 .* (x .* y .- z .* w),
        2 .* (x .* z .+ y .* w),
        2 .* (x .* y .+ z .* w),
        1 .- 2 .* (x.^2 .+ z.^2),
        2 .* (y .* z .- x .* w),
        2 .* (x .* z .- y .* w),
        2 .* (y .* z .+ x .* w),
        1 .- 2 .* (x.^2 .+ y.^2),
    )
end

"""
    make_matrix_svd_factors() -> Matrix{Float32}

Produce the (16, 9) factor matrix for the SVD quaternion method.
Entries encode the mapping from flattened 3×3 matrix to symmetric 4×4 matrix entries.
"""
function make_matrix_svd_factors()::Matrix{Float32}
    K = zeros(Float32, 16, 9)
    # This is the Davenport q-method / Shepperd's method factor matrix.
    # Encodes K = [K00,K01,...,K33] = factor_matrix @ [m00,m01,...,m22]
    # Row ordering: K[i,j] for (i,j) in row-major order
    # See: Farrell & Gratias (1992), Shuster (1993)
    K[1,1]  =  1f0; K[1,5]  =  1f0; K[1,9]  =  1f0   # K00 = m00+m11+m22
    K[6,1]  =  1f0; K[6,5]  = -1f0; K[6,9]  = -1f0   # K11 = m00-m11-m22
    K[11,1] = -1f0; K[11,5] =  1f0; K[11,9] = -1f0   # K22 = -m00+m11-m22
    K[16,1] = -1f0; K[16,5] = -1f0; K[16,9] =  1f0   # K33 = -m00-m11+m22
    K[2,6]  =  1f0; K[2,8]  =  1f0                    # K01 = m12+m21
    K[5,6]  =  1f0; K[5,8]  =  1f0                    # K10 = m12+m21
    K[3,3]  =  1f0; K[3,7]  =  1f0                    # K02 = m02+m20
    K[9,3]  =  1f0; K[9,7]  =  1f0                    # K20 = m02+m20
    K[4,2]  =  1f0; K[4,4]  =  1f0                    # K03 = m01+m10
    K[13,2] =  1f0; K[13,4] =  1f0                    # K30 = m01+m10
    K[7,6]  = -1f0; K[7,8]  =  1f0                    # K12 = m21-m12
    K[10,6] = -1f0; K[10,8] =  1f0                    # K21 = m21-m12
    K[8,3]  = -1f0; K[8,7]  =  1f0                    # K13 = m20-m02
    K[12,3] = -1f0; K[12,7] =  1f0                    # K31 = m20-m02
    K[14,2] = -1f0; K[14,4] =  1f0                    # K23 = m10-m01
    K[15,2] = -1f0; K[15,4] =  1f0                    # K32 = m10-m01
    return K
end

const MATRIX_SVD_QUAT_FACTORS = make_matrix_svd_factors()

"""
    largest_evec(m::AbstractMatrix{Float32}) -> Vector{Float32}

Return the eigenvector corresponding to the largest eigenvalue.
"""
function largest_evec(m::AbstractMatrix{Float32})::Vector{Float32}
    vals, vecs = eigen(Symmetric(m))
    idx = argmax(vals)
    return vecs[:, idx]
end

"""
    rot_from_svd(mat::AbstractArray; use_quat_formula=true) -> Rot3Array

Project an arbitrary 3×3 matrix to the nearest rotation matrix.
"""
function rot_from_svd(mat::AbstractArray; use_quat_formula::Bool=true)::Rot3Array
    @assert size(mat)[end-1:end] == (3,3) "Input must have last dims (3,3)"
    batch = size(mat)[1:end-2]

    if isempty(batch)
        # Single matrix case
        m = Float32.(mat)
        if use_quat_formula
            return _rot_from_svd_quat_single(m)
        else
            return _rot_from_svd_single(m)
        end
    end

    # Batch case: iterate
    result_fields = [Float32[] for _ in 1:9]
    for idx in CartesianIndices(batch)
        m = Float32.(mat[idx, :, :])
        r = use_quat_formula ? _rot_from_svd_quat_single(m) : _rot_from_svd_single(m)
        fields = (r.xx, r.xy, r.xz, r.yx, r.yy, r.yz, r.zx, r.zy, r.zz)
        for (k, f) in enumerate(fields)
            push!(result_fields[k], only(f))
        end
    end
    reshaped = [reshape(f, batch) for f in result_fields]
    return Rot3Array(reshaped...)
end

function _rot_from_svd_single(m::AbstractMatrix{Float32})::Rot3Array
    U, S, Vt = svd(m)
    d = sign(det(U * Vt'))
    diag_fix = Diagonal(Float32[1f0, 1f0, d])
    R = U * diag_fix * Vt'
    return rot_from_array(reshape(Float32.(R), 1, 1, 3, 3))
end

function _rot_from_svd_quat_single(m::AbstractMatrix{Float32})::Rot3Array
    # Flatten the 3×3 matrix to a length-9 vector
    mvec = vec(m')  # row-major
    # Compute the 16 entries of the symmetric 4×4 matrix
    K16 = MATRIX_SVD_QUAT_FACTORS * mvec  # (16,)
    K = reshape(K16, 4, 4)
    # Find largest eigenvector
    q = largest_evec(Float32.(Symmetric(K)))
    # q = [w, x, y, z]
    return rot_from_quaternion(q[1], q[2], q[3], q[4])
end

"""
    random_uniform_rotation(rng::AbstractRNG) -> Rot3Array

Sample a uniformly random rotation matrix from the Haar measure using a random quaternion.
"""
function random_uniform_rotation(rng::AbstractRNG)::Rot3Array
    # Sample from Haar measure via random quaternion
    q = randn(rng, Float32, 4)
    q ./= norm(q)
    w, x, y, z = q[1], q[2], q[3], q[4]
    return rot_from_quaternion(
        fill(w, ()), fill(x, ()), fill(y, ()), fill(z, ())
    )
end
