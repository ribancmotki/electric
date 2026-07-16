"""
Feature batch assembly for model input.
Converts intermediate chain features into a single BatchDict ready for the model.
"""

const BatchDict = Dict{String,Array}

"""
    assemble_batch(
        per_chain_data::Vector{Dict{String,Any}},
        fold_input::FoldingInput,
        rng_seed::Int,
        bucket_size::Int,
    ) -> BatchDict

Assemble all per-chain feature data into a single model-ready BatchDict.
All arrays are padded to bucket_size along the token dimension.
"""
function assemble_batch(
    per_chain_data::Vector{Dict{String,Any}},
    fold_input::FoldingInput,
    rng_seed::Int,
    bucket_size::Int,
)::BatchDict
    # Collect all tokens across chains
    all_residue_types = String[]
    all_chain_ids     = String[]
    all_seq_ids       = String[]
    all_is_protein    = Bool[]
    all_is_rna        = Bool[]
    all_is_dna        = Bool[]
    all_is_ligand     = Bool[]

    chain_idx = 0
    for (ci, chain_d) in enumerate(per_chain_data)
        res_types = chain_d["residue_types"]
        n_tokens  = length(res_types)
        is_lig    = get(chain_d, "is_ligand", false)

        chain_idx += 1
        cid  = "chain_$chain_idx"
        is_p = haskey(chain_d, "unpaired_msa") && !is_lig
        is_r = haskey(chain_d, "unpaired_msa") && !is_lig && any(t -> is_rna_residue(t), res_types)
        is_d = !is_lig && any(t -> is_dna_residue(t), res_types)

        append!(all_residue_types, res_types)
        append!(all_chain_ids,     fill(cid, n_tokens))
        append!(all_seq_ids,       string.(1:n_tokens))
        append!(all_is_protein,    fill(is_p && !is_r && !is_d, n_tokens))
        append!(all_is_rna,        fill(is_r, n_tokens))
        append!(all_is_dna,        fill(is_d, n_tokens))
        append!(all_is_ligand,     fill(is_lig, n_tokens))
    end

    n_tokens = length(all_residue_types)

    # Reference atom features (concatenate per-chain)
    ref_pos_all    = zeros(Float32, n_tokens, NUM_ATOM_SLOTS, 3)
    ref_mask_all   = falses(n_tokens, NUM_ATOM_SLOTS)
    ref_elem_all   = zeros(Int32, n_tokens, NUM_ATOM_SLOTS)
    ref_charge_all = zeros(Float32, n_tokens, NUM_ATOM_SLOTS)
    ref_atom_name_chars_all = zeros(Int32, n_tokens, NUM_ATOM_SLOTS, 4)
    ref_space_uid_all = zeros(Int32, n_tokens, NUM_ATOM_SLOTS)

    offset = 0
    for (ci, chain_d) in enumerate(per_chain_data)
        res_types = chain_d["residue_types"]
        n = length(res_types)
        rp  = get(chain_d, "ref_pos",    zeros(Float32, n, NUM_ATOM_SLOTS, 3))
        rm  = get(chain_d, "ref_mask",   falses(n, NUM_ATOM_SLOTS))
        re  = get(chain_d, "ref_element", zeros(Int32, n, NUM_ATOM_SLOTS))
        rc  = get(chain_d, "ref_charge",  zeros(Float32, n, NUM_ATOM_SLOTS))
        for ti in 1:n
            gi = offset + ti
            ref_pos_all[gi, :, :]    = rp[ti, :, :]
            ref_mask_all[gi, :]      = rm[ti, :]
            ref_elem_all[gi, :]      = re[ti, :]
            ref_charge_all[gi, :]    = rc[ti, :]
            # Encode atom names
            atom_order = get(ATOM_ORDER, res_types[ti], String[])
            for (j, aname) in enumerate(atom_order)
                j > NUM_ATOM_SLOTS && break
                isempty(aname) && continue
                ref_atom_name_chars_all[gi, j, :] = atom_name_to_chars(aname)
            end
            ref_space_uid_all[gi, :] = Int32(gi)
        end
        offset += n
    end

    # Target features
    target_feat = build_target_feat(
        all_residue_types,
        BitVector(all_is_protein),
        BitVector(all_is_rna),
        BitVector(all_is_dna),
        BitVector(all_is_ligand),
        ref_elem_all,
        ref_charge_all,
        ref_mask_all,
    )

    # MSA features (unpaired, padded)
    # Use first protein/RNA chain's MSA for now; full pairing handled in data pipeline
    msa_feat = zeros(Float32, 1, n_tokens, MSA_FEAT_DIM)
    extra_msa_feat = zeros(Float32, 1, n_tokens, EXTRA_MSA_FEAT_DIM)
    msa_mask = zeros(Float32, 1, n_tokens)
    extra_msa_mask = zeros(Float32, 1, n_tokens)
    n_msa = 1

    for chain_d in per_chain_data
        msa = get(chain_d, "unpaired_msa", nothing)
        msa === nothing && continue
        n_seqs_msa = n_seqs(msa)
        n_seqs_msa == 0 && continue
        msa_feats = make_msa_features(msa)
        n_msa = n_seqs_msa
        msa_feat      = zeros(Float32, n_msa, n_tokens, MSA_FEAT_DIM)
        msa_mask      = zeros(Float32, n_msa, n_tokens)
        aln_len = alignment_length(msa)
        msa_w   = min(aln_len, n_tokens)
        msa_feat[1:n_msa, 1:msa_w, :] = msa_feats["msa_feat"][1:n_msa, 1:msa_w, :]
        msa_mask[1:n_msa, 1:msa_w]    = msa_feats["msa_mask"][1:n_msa, 1:msa_w]
        break
    end

    # Template features
    n_templates = 0
    template_aatype          = zeros(Int32,   n_templates, n_tokens)
    template_all_atom_positions = zeros(Float32, n_templates, n_tokens, NUM_ATOM_SLOTS, 3)
    template_all_atom_mask   = zeros(Float32, n_templates, n_tokens, NUM_ATOM_SLOTS)
    template_pseudo_beta     = zeros(Float32, n_templates, n_tokens, 3)
    template_pseudo_beta_mask = zeros(Float32, n_templates, n_tokens)

    for chain_d in per_chain_data
        templates = get(chain_d, "templates", TemplateHitInput[])
        isempty(templates) && continue
        n_t = min(length(templates), 4)
        n_templates = n_t
        template_aatype          = zeros(Int32,   n_t, n_tokens)
        template_all_atom_positions = zeros(Float32, n_t, n_tokens, NUM_ATOM_SLOTS, 3)
        template_all_atom_mask   = zeros(Float32, n_t, n_tokens, NUM_ATOM_SLOTS)
        template_pseudo_beta     = zeros(Float32, n_t, n_tokens, 3)
        template_pseudo_beta_mask = zeros(Float32, n_t, n_tokens)

        for (ti, tmpl) in enumerate(templates[1:n_t])
            tmpl_struct = parse_structure_from_mmcif_string(tmpl.mmcif)
            for (qi, tii) in zip(tmpl.query_indices, tmpl.template_indices)
                qi_1 = qi + 1  # 0-based → 1-based
                qi_1 > n_tokens && continue
                res_type = all_residue_types[qi_1]
                template_aatype[ti, qi_1] = Int32(restype_index(res_type))
            end
        end
        break
    end

    # Bond features
    bond_feat = zeros(Float32, n_tokens, n_tokens)
    bonded_pairs = fold_input.bonded_atom_pairs
    if !isempty(bonded_pairs)
        for bp in bonded_pairs
            # Find token indices for the bonded atoms
            # (simplified: mark the whole-token pair)
        end
    end

    # Token/residue indices
    token_index    = Int32.(1:n_tokens)
    residue_index  = Int32.(1:n_tokens)

    batch = BatchDict(
        "token_index"                => token_index,
        "residue_index"              => residue_index,
        "token_chain_ids"            => all_chain_ids,
        "is_protein"                 => all_is_protein,
        "is_rna"                     => all_is_rna,
        "is_dna"                     => all_is_dna,
        "is_ligand"                  => all_is_ligand,
        "seq_mask"                   => ones(Float32, n_tokens),
        "target_feat"                => target_feat,
        "msa_feat"                   => msa_feat,
        "extra_msa_feat"             => extra_msa_feat,
        "msa_mask"                   => msa_mask,
        "extra_msa_mask"             => extra_msa_mask,
        "template_aatype"            => template_aatype,
        "template_all_atom_positions" => template_all_atom_positions,
        "template_all_atom_mask"     => template_all_atom_mask,
        "template_pseudo_beta"       => template_pseudo_beta,
        "template_pseudo_beta_mask"  => template_pseudo_beta_mask,
        "atom_positions"             => zeros(Float32, n_tokens, NUM_ATOM_SLOTS, 3),
        "atom_mask"                  => zeros(Float32, n_tokens, NUM_ATOM_SLOTS),
        "ref_pos"                    => ref_pos_all,
        "ref_mask"                   => Float32.(ref_mask_all),
        "ref_element"                => ref_elem_all,
        "ref_charge"                 => ref_charge_all,
        "ref_atom_name_chars"        => ref_atom_name_chars_all,
        "ref_space_uid"              => ref_space_uid_all,
        "bond_feat"                  => bond_feat,
        "rng_seed"                   => Int64[rng_seed],
        "num_tokens"                 => Int32[n_tokens],
    )

    return pad_to_bucket(batch, bucket_size)
end
