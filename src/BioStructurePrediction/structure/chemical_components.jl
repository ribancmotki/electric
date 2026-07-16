"""
Structure-level chemical component utilities.
Re-exports and extends CCD lookups for use with Structure objects.
"""

"""
    get_residue_component_atoms(s::Structure, ccd::Ccd, chain_id::String, seq_id::String) -> NamedTuple

Return CCD atom data for the residue at (chain_id, seq_id) in the structure.
"""
function get_residue_component_atoms(s::Structure, ccd::Ccd, chain_id::String, seq_id::String)::NamedTuple
    comp_ids = get_column(s.atoms, :label_comp_id)
    chain_ids = get_column(s.atoms, :label_asym_id)
    seq_ids   = get_column(s.atoms, :label_seq_id)

    # Find the comp_id for this residue
    idx = findfirst(i -> chain_ids[i] == chain_id && string(seq_ids[i]) == seq_id, 1:nrows(s.atoms))
    idx === nothing && error("Residue not found: chain=$chain_id seq=$seq_id")
    comp_id = comp_ids[idx]
    return get_component_atoms(ccd, comp_id)
end

"""
    compute_chirality_sign(pos_center::Vector{Float32}, pos_n1::Vector{Float32},
                           pos_n2::Vector{Float32}, pos_n3::Vector{Float32}) -> Float32

Compute the sign of the chiral determinant for a chiral center.
Returns +1.0 for one handedness, -1.0 for the other.
"""
function compute_chirality_sign(
    pos_center::Vector{Float32},
    pos_n1::Vector{Float32},
    pos_n2::Vector{Float32},
    pos_n3::Vector{Float32}
)::Float32
    v1 = pos_n1 - pos_center
    v2 = pos_n2 - pos_center
    v3 = pos_n3 - pos_center
    det = v1[1] * (v2[2]*v3[3] - v2[3]*v3[2]) -
          v1[2] * (v2[1]*v3[3] - v2[3]*v3[1]) +
          v1[3] * (v2[1]*v3[2] - v2[2]*v3[1])
    return sign(det)
end

"""
    compare_chirality(predicted_structure::Structure, reference_structure::Structure, ccd::Ccd) -> Dict

Compare chirality of chiral centers in the predicted structure against the reference (CCD ideal geometry).
Returns a Dict with keys:
- n_correct: number of correctly oriented chiral centers
- n_inverted: number of inverted chiral centers
- n_missing: number of chiral centers that could not be evaluated
- fraction_correct: n_correct / (n_correct + n_inverted)
"""
function compare_chirality(
    predicted_structure::Structure,
    reference_structure::Structure,
    ccd::Ccd
)::Dict{String,Any}
    n_correct  = 0
    n_inverted = 0
    n_missing  = 0

    pred_chains = get_column(predicted_structure.atoms, :label_asym_id)
    pred_seqs   = get_column(predicted_structure.atoms, :label_seq_id)
    pred_comps  = get_column(predicted_structure.atoms, :label_comp_id)
    pred_atoms  = get_column(predicted_structure.atoms, :label_atom_id)
    pred_xs     = get_column(predicted_structure.atoms, :Cartn_x)
    pred_ys     = get_column(predicted_structure.atoms, :Cartn_y)
    pred_zs     = get_column(predicted_structure.atoms, :Cartn_z)

    # Build atom position index for predicted structure
    pred_index = Dict{Tuple{String,String,String},Vector{Float32}}()
    for i in 1:num_atoms(predicted_structure)
        key = (pred_chains[i], string(pred_seqs[i]), pred_atoms[i])
        pred_index[key] = Float32[pred_xs[i], pred_ys[i], pred_zs[i]]
    end

    # Get all residues with chiral centers
    for residue in predicted_structure.residues
        comp_id = residue.comp_id
        centers = get_chiral_centers(comp_id)
        isempty(centers) && continue

        cid = residue.chain_id
        sid = residue.seq_id

        # Get reference chirality from CCD
        comp = get_component(ccd, comp_id)
        comp === nothing && continue

        for (center_atom, neighbor_atoms) in centers
            length(neighbor_atoms) < 3 && continue
            pos_center = get(pred_index, (cid, sid, center_atom), nothing)
            pos_n1     = get(pred_index, (cid, sid, neighbor_atoms[1]), nothing)
            pos_n2     = get(pred_index, (cid, sid, neighbor_atoms[2]), nothing)
            pos_n3     = get(pred_index, (cid, sid, neighbor_atoms[3]), nothing)

            if any(isnothing, [pos_center, pos_n1, pos_n2, pos_n3])
                n_missing += 1
                continue
            end

            pred_sign = compute_chirality_sign(pos_center, pos_n1, pos_n2, pos_n3)

            # Get reference positions from CCD ideal coordinates
            ccd_atoms = get_component_atoms(ccd, comp_id)
            atom_name_to_idx = Dict(ccd_atoms.atom_names[i] => i for i in eachindex(ccd_atoms.atom_names))
            center_idx = get(atom_name_to_idx, center_atom, nothing)
            n1_idx     = get(atom_name_to_idx, neighbor_atoms[1], nothing)
            n2_idx     = get(atom_name_to_idx, neighbor_atoms[2], nothing)
            n3_idx     = get(atom_name_to_idx, neighbor_atoms[3], nothing)

            if any(isnothing, [center_idx, n1_idx, n2_idx, n3_idx])
                n_missing += 1
                continue
            end

            ref_center = ccd_atoms.ideal_pos[center_idx, :]
            ref_n1     = ccd_atoms.ideal_pos[n1_idx, :]
            ref_n2     = ccd_atoms.ideal_pos[n2_idx, :]
            ref_n3     = ccd_atoms.ideal_pos[n3_idx, :]

            ref_sign = compute_chirality_sign(ref_center, ref_n1, ref_n2, ref_n3)

            if ref_sign == 0f0 || pred_sign == 0f0
                n_missing += 1
            elseif pred_sign == ref_sign
                n_correct += 1
            else
                n_inverted += 1
            end
        end
    end

    fraction_correct = (n_correct + n_inverted) > 0 ?
        Float64(n_correct) / Float64(n_correct + n_inverted) : 1.0

    return Dict{String,Any}(
        "n_correct"        => n_correct,
        "n_inverted"       => n_inverted,
        "n_missing"        => n_missing,
        "fraction_correct" => fraction_correct,
    )
end
