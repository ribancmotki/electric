"""
feat_batch.jl — Batch assembly from per-chain features.
"""

using Random
using Logging

"""
    assemble_batch(fold_input::Input, ccd::Ccd, rng::AbstractRNG;
                   max_msa_seqs::Int=512,
                   max_extra_msa_seqs::Int=1024,
                   max_template_hits::Int=4,
                   max_atoms_per_token::Int=24,
                   with_hydrogens::Bool=false) -> Dict{String,Array}

Assemble a full model input batch dict from a folding input.
"""
function assemble_batch(
    fold_input::Input,
    ccd::Ccd,
    rng::AbstractRNG;
    max_msa_seqs::Int       = 512,
    max_extra_msa_seqs::Int = 1024,
    max_template_hits::Int  = 4,
    max_atoms_per_token::Int = 24,
    with_hydrogens::Bool    = false,
)::Dict{String,Any}
    # Build residues + flat atom layout
    residues = _build_residues(fold_input, ccd)
    flat_layout = make_flat_atom_layout(residues, ccd;
                                        with_hydrogens=with_hydrogens)

    # Build token layout
    all_tokens, all_token_atoms, standard_token_idxs = tokenizer(
        flat_layout, ccd, max_atoms_per_token)

    num_tokens = length(all_tokens)
    @debug "assemble_batch: $(length(fold_input.chains)) chains, $num_tokens tokens"

    batch = Dict{String,Any}()

    # ── Token features ──────────────────────────────────────────────────────
    token_feats = compute_token_features(fold_input, flat_layout, all_tokens)
    batch["token_features"] = token_feats

    # ── Reference structure ─────────────────────────────────────────────────
    ref_struct_feats = compute_ref_structure(all_tokens, all_token_atoms, ccd)
    for (k, v) in ref_struct_feats
        batch[k] = v
    end

    # ── MSA features (concat all chains) ────────────────────────────────────
    all_msa         = Int8[]
    all_del_mat     = Int8[]
    all_msa_mask    = Bool[]
    msa_chain_ids   = Int32[]

    for chain in fold_input.chains
        (chain isa ProteinChain || chain isa RnaChain || chain isa DnaChain) || continue
        chain_toks = _token_indices_for_chain(all_tokens, chain.id)
        isempty(chain_toks) && continue
        n_chain_toks = length(chain_toks)

        mf = compute_msa_features_for_chain(chain, n_chain_toks;
                                             max_msa_seqs=max_msa_seqs)
        # Store per-chain MSA — full batch MSA assembly done in merging_features
        batch["chain_msa_$(chain.id)"] = mf
    end

    # ── Template features ────────────────────────────────────────────────────
    template_feats = _compute_template_features(fold_input, all_tokens,
                                                 all_token_atoms, ccd,
                                                 max_template_hits)
    batch["template_features"] = template_feats

    # ── Atom layout metadata ─────────────────────────────────────────────────
    batch["flat_layout"]          = flat_layout
    batch["all_tokens"]           = all_tokens
    batch["all_token_atoms"]      = all_token_atoms
    batch["standard_token_idxs"]  = standard_token_idxs
    batch["num_tokens"]           = num_tokens

    # ── Pair features (relative encoding) ───────────────────────────────────
    rel_enc = create_relative_encoding(token_feats)
    batch["pair_relative_encoding"] = rel_enc

    return batch
end

function _build_residues(fold_input::Input, ccd::Ccd)::Residues
    res_names  = String[]
    res_ids    = Int[]
    chain_ids  = String[]
    chain_types = String[]
    is_start   = Bool[]
    is_end     = Bool[]
    smiles_strs = Union{String,Nothing}[]

    for chain in fold_input.chains
        ct = _chain_poly_type_or_ligand(chain)
        res_seq = _chain_residues(chain, ccd)
        n = length(res_seq)
        for (j, (rid, rn)) in enumerate(res_seq)
            push!(res_names,   rn)
            push!(res_ids,     rid)
            push!(chain_ids,   chain.id)
            push!(chain_types, ct)
            push!(is_start,    j == 1)
            push!(is_end,      j == n)
            push!(smiles_strs, _get_smiles(chain, j))
        end
    end

    return Residues(res_names, res_ids, chain_ids, chain_types,
                    is_start, is_end, nothing, smiles_strs)
end

function _chain_poly_type_or_ligand(chain::ProteinChain)::String = PROTEIN_CHAIN
function _chain_poly_type_or_ligand(chain::RnaChain)::String     = RNA_CHAIN
function _chain_poly_type_or_ligand(chain::DnaChain)::String     = DNA_CHAIN
function _chain_poly_type_or_ligand(chain::Ligand)::String       = LIGAND_CHAIN

function _chain_residues(chain::ProteinChain, ccd::Ccd)::Vector{Tuple{Int,String}}
    residues = to_ccd_sequence(chain)
    return [(i, rn) for (i, rn) in enumerate(residues)]
end

function _chain_residues(chain::RnaChain, ccd::Ccd)::Vector{Tuple{Int,String}}
    residues = to_ccd_sequence(chain)
    return [(i, rn) for (i, rn) in enumerate(residues)]
end

function _chain_residues(chain::DnaChain, ccd::Ccd)::Vector{Tuple{Int,String}}
    residues = to_ccd_sequence(chain)
    return [(i, rn) for (i, rn) in enumerate(residues)]
end

function _chain_residues(chain::Ligand, ccd::Ccd)::Vector{Tuple{Int,String}}
    return [(1, chain.ccd_id)]
end

function _get_smiles(chain::Ligand, j::Int)::Union{String,Nothing}
    return chain.smiles
end
function _get_smiles(chain, j::Int)::Union{String,Nothing}
    return nothing
end

function _token_indices_for_chain(all_tokens::AtomLayout, chain_id::String)::Vector{Int}
    return findall(i -> all_tokens.chain_id[i] == chain_id, 1:length(all_tokens))
end

function _compute_template_features(
    fold_input::Input,
    all_tokens::AtomLayout,
    all_token_atoms::AtomLayout,
    ccd::Ccd,
    max_template_hits::Int,
)::Dict{String,Array}
    num_tokens = length(all_tokens)

    # Collect templates from protein chains only
    all_template_aatype         = Array{Int32}[]
    all_template_atom_positions = Array{Float32}[]
    all_template_atom_mask      = Array{Bool}[]

    for chain in fold_input.chains
        chain isa ProteinChain || continue
        isempty(chain.templates) && continue

        chain_toks = _token_indices_for_chain(all_tokens, chain.id)
        n_chain_toks = length(chain_toks)

        for tmpl in chain.templates[1:min(max_template_hits, length(chain.templates))]
            if length(all_template_aatype) >= max_template_hits
                break
            end

            aatype_t = zeros(Int32, num_tokens)
            pos_t    = zeros(Float32, num_tokens, 37, 3)
            mask_t   = falses(num_tokens, 37)

            # Parse template structure
            tmpl_struct = try
                parse_structure_from_mmcif_string(tmpl.mmcif_str)
            catch
                continue
            end

            # Fill in template features for chain tokens
            mapping = tmpl.query_to_template_map
            tmpl_feats = get_polymer_features(tmpl_struct, PROTEIN_CHAIN,
                                               n_chain_toks, mapping)

            for (local_i, global_i) in enumerate(chain_toks)
                aatype_t[global_i]        = get(tmpl_feats["template_aatype"], local_i, Int32(0))
                if local_i <= size(tmpl_feats["template_atom_positions"], 1)
                    pos_t[global_i, :, :]  = tmpl_feats["template_atom_positions"][local_i, 1:min(37, end), :]
                    mask_t[global_i, :]    = tmpl_feats["template_atom_mask"][local_i, 1:min(37, end)]
                end
            end

            push!(all_template_aatype,         aatype_t)
            push!(all_template_atom_positions,  pos_t)
            push!(all_template_atom_mask,       mask_t)
        end
    end

    # Pad to max_template_hits
    n_tmpl = length(all_template_aatype)
    for _ in (n_tmpl+1):max_template_hits
        push!(all_template_aatype,         zeros(Int32,   num_tokens))
        push!(all_template_atom_positions,  zeros(Float32, num_tokens, 37, 3))
        push!(all_template_atom_mask,       falses(num_tokens, 37))
    end

    aatype_stacked = zeros(Int32, max_template_hits, num_tokens)
    pos_stacked    = zeros(Float32, max_template_hits, num_tokens, 37, 3)
    mask_stacked   = falses(max_template_hits, num_tokens, 37)
    tmpl_mask      = vcat(fill(true, n_tmpl), fill(false, max_template_hits - n_tmpl))

    for i in 1:max_template_hits
        aatype_stacked[i, :]      = all_template_aatype[i]
        pos_stacked[i, :, :, :]  = all_template_atom_positions[i]
        mask_stacked[i, :, :]    = all_template_atom_mask[i]
    end

    return Dict{String,Array}(
        "template_aatype"              => aatype_stacked,
        "template_all_atom_positions"  => pos_stacked,
        "template_all_atom_mask"       => mask_stacked,
        "template_mask"                => tmpl_mask,
    )
end
