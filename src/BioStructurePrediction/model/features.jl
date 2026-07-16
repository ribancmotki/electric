"""
Feature construction utilities for building model input tensors.
"""

# ──────────────────────────────────────────────
#  Target feature encoding (447 dims)
# ──────────────────────────────────────────────

"""
    build_target_feat(
        token_residue_types::Vector{String},
        token_is_protein::BitVector,
        token_is_rna::BitVector,
        token_is_dna::BitVector,
        token_is_ligand::BitVector,
        ref_elem::Matrix{Int32},
        ref_charge::Matrix{Float32},
        ref_mask::Matrix{Bool},
    ) -> Matrix{Float32}

Build the 447-dimensional target feature tensor.
Shape: (num_tokens, 447)

Encoding layout:
   0-31:   residue type one-hot (32 dims)
  32-33:   is_protein, is_rna, is_dna, is_ligand (4 dims)
  34-37:   entity type bits (4 dims — reserved)
  38-165:  per-slot element one-hot (36 slots × ~4 dim compressed, simplified here)
  ...      remaining dims from physicochemical features
"""
function build_target_feat(
    token_residue_types::Vector{String},
    token_is_protein::BitVector,
    token_is_rna::BitVector,
    token_is_dna::BitVector,
    token_is_ligand::BitVector,
    ref_elem::AbstractMatrix{Int32},
    ref_charge::AbstractMatrix{Float32},
    ref_mask::AbstractMatrix{Bool},
)::Matrix{Float32}
    n = length(token_residue_types)
    feat = zeros(Float32, n, TARGET_FEAT_DIM)

    for i in 1:n
        # Residue type one-hot (dims 1-32)
        rt_idx = restype_index(token_residue_types[i])
        rt_idx_1based = rt_idx + 1
        if 1 <= rt_idx_1based <= NUM_RESIDUE_TYPES
            feat[i, rt_idx_1based] = 1f0
        end

        # Entity type flags (dims 33-36)
        feat[i, 33] = token_is_protein[i] ? 1f0 : 0f0
        feat[i, 34] = token_is_rna[i]     ? 1f0 : 0f0
        feat[i, 35] = token_is_dna[i]     ? 1f0 : 0f0
        feat[i, 36] = token_is_ligand[i]  ? 1f0 : 0f0

        # Per-slot element encoding (dims 37-292, one-hot over 64 element types per slot, simplified)
        # We use a compressed encoding: 8 dims per slot × 32 slots = 256 dims
        offset = 36
        for j in 1:min(NUM_ATOM_SLOTS, 32)
            j > size(ref_elem, 2) && break
            ref_mask[i, j] || continue
            elem_idx = Int(ref_elem[i, j])
            if 1 <= elem_idx <= 8
                feat[i, offset + (j-1)*8 + elem_idx] = 1f0
            end
        end

        # Charge features (dims 293-328, one per slot)
        charge_offset = 292
        for j in 1:min(NUM_ATOM_SLOTS, 36)
            j > size(ref_charge, 2) && break
            feat[i, charge_offset + j] = clamp(ref_charge[i, j], -5f0, 5f0) / 5f0
        end

        # Atom count feature (dim 329)
        n_atoms_present = sum(ref_mask[i, :])
        feat[i, 329] = Float32(n_atoms_present) / Float32(NUM_ATOM_SLOTS)

        # Polymer/ligand composition (dims 330-340)
        feat[i, 330] = is_standard_amino_acid(token_residue_types[i]) ? 1f0 : 0f0
        feat[i, 331] = is_rna_residue(token_residue_types[i]) ? 1f0 : 0f0
        feat[i, 332] = is_dna_residue(token_residue_types[i]) ? 1f0 : 0f0

        # Remaining dims (341-447): zeros by default
        # These would encode additional physicochemical properties in a full implementation
    end
    return feat
end

# ──────────────────────────────────────────────
#  Template pair features (39 dims)
# ──────────────────────────────────────────────

"""
    build_template_pair_features(
        template_positions::AbstractArray{Float32,4},  # (n_templates, n_tokens, 37, 3)
        template_mask::AbstractArray{Float32,3},        # (n_templates, n_tokens, 37)
        query_positions::AbstractArray{Float32,3},      # (n_tokens, 37, 3)
        query_mask::AbstractMatrix{Float32},            # (n_tokens, 37)
    ) -> Array{Float32,4}  # (n_templates, n_tokens, n_tokens, 39)

Compute template pair features encoding inter-residue distances and angles.
"""
function build_template_pair_features(
    template_positions::AbstractArray{Float32,4},
    template_mask::AbstractArray{Float32,3},
    query_positions::AbstractArray{Float32,3},
    query_mask::AbstractMatrix{Float32},
)::Array{Float32,4}
    n_templates, n_tokens, _, _ = size(template_positions)
    n_bins = TEMPLATE_FEAT_DIM  # 39

    pair_feats = zeros(Float32, n_templates, n_tokens, n_tokens, n_bins)

    for t in 1:n_templates
        for i in 1:n_tokens, j in 1:n_tokens
            i == j && continue
            # Use Cβ atom (slot 5) or Cα (slot 2) for distance
            cb_slot = 5
            ca_slot = 2

            pos_i_cb = template_mask[t, i, cb_slot] > 0f0 ?
                Float32.(template_positions[t, i, cb_slot, :]) : nothing
            pos_j_cb = template_mask[t, j, cb_slot] > 0f0 ?
                Float32.(template_positions[t, j, cb_slot, :]) : nothing
            pos_i_ca = template_mask[t, i, ca_slot] > 0f0 ?
                Float32.(template_positions[t, i, ca_slot, :]) : nothing
            pos_j_ca = template_mask[t, j, ca_slot] > 0f0 ?
                Float32.(template_positions[t, j, ca_slot, :]) : nothing

            # Use CB if available, else CA
            pi = something(pos_i_cb, pos_i_ca, nothing)
            pj = something(pos_j_cb, pos_j_ca, nothing)

            if pi !== nothing && pj !== nothing
                dist = sqrt(sum((pi .- pj).^2))
                # Bin the distance (0-22 Å in 36 bins)
                bin_idx = min(Int(floor(dist / 22f0 * 36f0)) + 1, 36)
                pair_feats[t, i, j, bin_idx] = 1f0
                pair_feats[t, i, j, 37] = 1f0  # both atoms present
            end

            # Same chain indicator (bin 38-39)
            pair_feats[t, i, j, 38] = 1f0  # always same template
        end
    end

    return pair_feats
end
