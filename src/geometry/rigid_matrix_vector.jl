"""
rigid_matrix_vector.jl — Rigid3Array: SE(3) rigid transforms.
"""

using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# Rigid3Array
# ──────────────────────────────────────────────────────────────────────────────

"""
    Rigid3Array

Batch of rigid transforms (rotation + translation).
- `rotation::Rot3Array` — rotation component.
- `translation::Vec3Array` — translation component.
"""
struct Rigid3Array
    rotation::Rot3Array
    translation::Vec3Array
end

function Base.show(io::IO, r::Rigid3Array)
    print(io, "Rigid3Array$(size(r.rotation))")
end

Base.size(r::Rigid3Array) = size(r.rotation)

function Base.getindex(r::Rigid3Array, idxs...)
    return Rigid3Array(r.rotation[idxs...], r.translation[idxs...])
end

# ──────────────────────────────────────────────────────────────────────────────
# Operations
# ──────────────────────────────────────────────────────────────────────────────

"""
    rigid_compose(a::Rigid3Array, b::Rigid3Array) -> Rigid3Array

Compose two rigid transforms: T = A ∘ B.
"""
function rigid_compose(a::Rigid3Array, b::Rigid3Array)::Rigid3Array
    new_rot   = rot_compose(a.rotation, b.rotation)
    new_trans = rot_apply_to_point(a.rotation, b.translation) + a.translation
    return Rigid3Array(new_rot, new_trans)
end

Base.:*(a::Rigid3Array, b::Rigid3Array) = rigid_compose(a, b)

"""
    rigid_inverse(r::Rigid3Array) -> Rigid3Array

Compute the inverse of a rigid transform.
"""
function rigid_inverse(r::Rigid3Array)::Rigid3Array
    inv_rot   = rot_inverse(r.rotation)
    inv_trans = -rot_apply_to_point(inv_rot, r.translation)
    return Rigid3Array(inv_rot, inv_trans)
end

"""
    rigid_apply_to_point(r::Rigid3Array, p::Vec3Array) -> Vec3Array

Apply rigid transform to a point.
"""
function rigid_apply_to_point(r::Rigid3Array, p::Vec3Array)::Vec3Array
    return rot_apply_to_point(r.rotation, p) + r.translation
end

"""
    rigid_apply_inverse_to_point(r::Rigid3Array, p::Vec3Array) -> Vec3Array

Apply the inverse rigid transform to a point.
"""
function rigid_apply_inverse_to_point(r::Rigid3Array, p::Vec3Array)::Vec3Array
    return rot_apply_inverse_to_point(r.rotation, p - r.translation)
end

"""
    compose_rotation(r::Rigid3Array, extra_rot::Rot3Array) -> Rigid3Array

Compose with an additional rotation (applied to the right of the existing rotation).
"""
function compose_rotation(r::Rigid3Array, extra_rot::Rot3Array)::Rigid3Array
    return Rigid3Array(rot_compose(r.rotation, extra_rot), r.translation)
end

"""
    scale_translation(r::Rigid3Array, factor) -> Rigid3Array

Scale the translation component.
"""
function scale_translation(r::Rigid3Array, factor)::Rigid3Array
    return Rigid3Array(r.rotation, factor * r.translation)
end

# ──────────────────────────────────────────────────────────────────────────────
# Constructors
# ──────────────────────────────────────────────────────────────────────────────

"""
    rigid_identity(shape::Tuple; dtype=Float32) -> Rigid3Array

Identity rigid transform for a given batch shape.
"""
function rigid_identity(shape::Tuple; dtype::Type=Float32)::Rigid3Array
    return Rigid3Array(
        rot_identity(shape; dtype),
        vec3_zeros(shape; dtype),
    )
end

rigid_identity(shape::Int...) = rigid_identity(shape)

"""
    rigid_from_array(arr::AbstractArray) -> Rigid3Array

Construct from array of shape (..., 3, 4) where last 3 columns are translation.
"""
function rigid_from_array(arr::AbstractArray)::Rigid3Array
    @assert size(arr)[end-1:end] == (3,4) "Last dims must be (3,4)"
    batch = size(arr)[1:end-2]
    idxs  = ntuple(i -> Colon(), length(batch))
    rot_arr = arr[idxs..., :, 1:3]
    rot = rot_from_array(rot_arr)
    tx = Float32.(arr[idxs..., 1, 4])
    ty = Float32.(arr[idxs..., 2, 4])
    tz = Float32.(arr[idxs..., 3, 4])
    return Rigid3Array(rot, Vec3Array(tx, ty, tz))
end

"""
    rigid_to_array(r::Rigid3Array) -> Array{Float32}

Convert to array of shape (..., 3, 4).
"""
function rigid_to_array(r::Rigid3Array)::Array{Float32}
    rot_arr = rot_to_array(r.rotation)  # (..., 3, 3)
    sh = size(r.translation.x)
    result = zeros(Float32, sh..., 3, 4)
    idxs = ntuple(i -> Colon(), length(sh))
    result[idxs..., :, 1:3] = rot_arr
    result[idxs..., 1, 4] = r.translation.x
    result[idxs..., 2, 4] = r.translation.y
    result[idxs..., 3, 4] = r.translation.z
    return result
end

"""
    rigid_from_array4x4(arr::AbstractArray) -> Rigid3Array

Construct from array of shape (..., 4, 4).
"""
function rigid_from_array4x4(arr::AbstractArray)::Rigid3Array
    @assert size(arr)[end-1:end] == (4,4) "Last dims must be (4,4)"
    batch = size(arr)[1:end-2]
    idxs  = ntuple(i -> Colon(), length(batch))
    rot_arr = arr[idxs..., 1:3, 1:3]
    rot = rot_from_array(rot_arr)
    tx = Float32.(arr[idxs..., 1, 4])
    ty = Float32.(arr[idxs..., 2, 4])
    tz = Float32.(arr[idxs..., 3, 4])
    return Rigid3Array(rot, Vec3Array(tx, ty, tz))
end

# ──────────────────────────────────────────────────────────────────────────────
# Point alignment (Kabsch/SVD superposition)
# ──────────────────────────────────────────────────────────────────────────────

"""
    _compute_covariance_matrix(row_values::Vec3Array, col_values::Vec3Array,
                               weights::AbstractArray; epsilon=1f-6) -> Array{Float32}

Compute a weighted covariance matrix of shape (..., 3, 3).
"""
function _compute_covariance_matrix(row_values::Vec3Array, col_values::Vec3Array,
                                    weights::AbstractArray; epsilon::Float32=1f-6)
    # Flatten: (n, 3) shape vectors
    r_arr = vec3_to_array(row_values)  # (..., n, 3)
    c_arr = vec3_to_array(col_values)  # (..., n, 3)
    w_sum = sum(weights) + epsilon

    # Weighted mean subtraction
    rm = sum(weights .* r_arr, dims=ndims(r_arr)-1) ./ w_sum
    cm = sum(weights .* c_arr, dims=ndims(c_arr)-1) ./ w_sum
    r_centered = r_arr .- rm
    c_centered = c_arr .- cm

    # Weighted covariance: C = sum_i w_i * r_i^T * c_i
    # For 1D batch: C[a,b] = sum_i w_i * r_i[a] * c_i[b]
    w_r = weights .* r_centered  # (..., n, 3)
    # einsum("...na,...nb->...ab")
    n_atoms = size(r_centered, ndims(r_centered)-1)
    batch_sh = size(r_centered)[1:end-2]
    cov = zeros(Float32, batch_sh..., 3, 3)
    for a in 1:3, b in 1:3
        cov_ab = sum(w_r[ntuple(i->Colon(),length(batch_sh))..., :, a] .*
                     c_centered[ntuple(i->Colon(),length(batch_sh))..., :, b],
                     dims=ndims(c_centered)-1)
        cov[ntuple(i->Colon(),length(batch_sh))..., a, b] = dropdims(cov_ab, dims=ndims(cov_ab))
    end
    return cov
end

"""
    rigid_from_point_alignment(points_to::Vec3Array, points_from::Vec3Array;
                               weights=nothing, epsilon=1f-6) -> Rigid3Array

Find the optimal rigid transform aligning `points_from` → `points_to` using weighted SVD.
"""
function rigid_from_point_alignment(points_to::Vec3Array, points_from::Vec3Array;
                                    weights=nothing, epsilon::Float32=1f-6)::Rigid3Array
    n = length(points_to.x)
    w = weights === nothing ? fill(1f0, n) : Float32.(weights)
    w_sum = sum(w) + epsilon

    # Weighted centroids
    to_arr   = vec3_to_array(points_to)    # (n, 3)
    from_arr = vec3_to_array(points_from)  # (n, 3)

    to_center   = dropdims(sum(w .* to_arr,   dims=1), dims=1) ./ w_sum  # (3,)
    from_center = dropdims(sum(w .* from_arr, dims=1), dims=1) ./ w_sum  # (3,)

    to_centered   = to_arr   .- to_center'
    from_centered = from_arr .- from_center'

    # Covariance: H = from^T * W * to (3×3)
    H = (w .* from_centered)' * to_centered  # (3×3)

    # SVD
    Usvd, _, Vt = svd(H)
    d = sign(det(Vt' * Usvd'))
    diag_fix = Diagonal(Float32[1f0, 1f0, d])
    R = Vt' * diag_fix * Usvd'  # 3×3 rotation matrix

    rot = rot_from_array(reshape(Float32.(R), 1, 1, 3, 3))

    # Translation: t = to_center - R * from_center
    t_from = Vec3Array(
        fill(from_center[1], ()),
        fill(from_center[2], ()),
        fill(from_center[3], ()),
    )
    r_from = rot_apply_to_point(rot, t_from)
    t = Vec3Array(
        fill(to_center[1], ()) .- r_from.x,
        fill(to_center[2], ()) .- r_from.y,
        fill(to_center[3], ()) .- r_from.z,
    )
    return Rigid3Array(rot, t)
end
