"""
features.jl — Feature structs and compute_features methods.
"""

using Random

# ──────────────────────────────────────────────────────────────────────────────
# Feature structs
# ──────────────────────────────────────────────────────────────────────────────

"""
    TokenFeatures

Per-token identity and position features.
"""
struct TokenFeatures
    aatype::Array{Int32}            # (num_tokens,)
    residue_index::Array{Int32}     # (num_tokens,)
    token_index::Array{Int32}       # (num_tokens,)
    asym_id::Array{Int32}           # (num_tokens,)
    entity_id::Array{Int32}         # (num_tokens,)
    sym_id::Array{Int32}            # (num_tokens,)
    is_protein::Array{Bool}         # (num_tokens,)
    is_rna::Array{Bool}             # (num_tokens,)
    is_dna::Array{Bool}             # (num_tokens,)
    is_ligand::Array{Bool}          # (num_tokens,)
    token_mask::Array{Bool}         # (num_tokens,)
    num_sym::Array{Int32}           # (num_tokens,)
end

"""
    MSAFeatures

MSA-related features for a single chain.
"""
struct MSAFeatures
    msa::Array{Int8}                # (num_seqs, num_tokens)
    deletion_matrix::Array{Int8}    # (num_seqs, num_tokens)
    msa_mask::Array{Bool}           # (num_seqs, num_tokens)
    cluster_bias_mask::Array{Bool}  # (num_seqs,)
    msa_profile::Array{Float32}     # (num_tokens, num_residue_types)
end

"""
    TemplateFeatures

Template-derived structural features.
"""
struct TemplateFeatures
    template_aatype::Array{Int32}         # (num_templates, num_tokens)
    template_all_atom_positions::Array{Float32}  # (num_templates, num_tokens, 37, 3)
    template_all_atom_mask::Array{Bool}   # (num_templates, num_tokens, 37)
    template_mask::Array{Bool}            # (num_templates,)
end

"""
    RefStructure

Reference structure for atom-level conditioning.
"""
struct RefStructure
    ref_pos::Array{Float32}         # (num_tokens, max_atoms, 3)
    ref_mask::Array{Bool}           # (num_tokens, max_atoms)
    ref_element::Array{Int32}       # (num_tokens, max_atoms) — 0-based element index
    ref_charge::Array{Float32}      # (num_tokens, max_atoms)
    ref_atom_name_chars::Array{Int32} # (num_tokens, max_atoms, 4)
    ref_space_uid::Array{Int32}     # (num_tokens, max_atoms)
end

"""
    PredictedStructureInfo

Contains predicted coordinates and masks, updated per recycle.
"""
mutable struct PredictedStructureInfo
    positions::Array{Float32}       # (num_tokens, max_atoms, 3)
    mask::Array{Bool}               # (num_tokens, max_atoms)
    b_factors::Array{Float32}       # (num_tokens, max_atoms)
end

"""
    AtomCrossAttFeatures

Atom cross-attention encoder outputs.
"""
struct AtomCrossAttFeatures
    token_single::Array{Float32}    # (num_tokens, c_s)
    token_pair::Array{Float32}      # (num_tokens, num_tokens, c_z)
end

"""
    ConvertModelOutput

Output from the structure module.
"""
struct ConvertModelOutput
    final_atom_positions::Array{Float32}  # (num_tokens, max_atoms, 3)
    final_atom_mask::Array{Bool}          # (num_tokens, max_atoms)
    final_frames::Array{Float32}          # (num_tokens, 4, 4)
end

"""
    Frames

Backbone rigid frames (transforms) per token.
"""
struct Frames
    rotation::Array{Float32}        # (num_tokens, 3, 3)
    translation::Array{Float32}     # (num_tokens, 3)
end

# ──────────────────────────────────────────────────────────────────────────────
# Batch dict type alias
# ──────────────────────────────────────────────────────────────────────────────

const BatchDictFull = Dict{String,Any}

# ──────────────────────────────────────────────────────────────────────────────
# compute_features — token features from folding input
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_token_features(fold_input::Input, flat_layout::AtomLayout,
                            all_tokens::AtomLayout) -> Dict{String,Array}
"""
function compute_token_features(
    fold_input::Input,
    flat_layout::AtomLayout,
    all_tokens::AtomLayout,
)::Dict{String,Array}
    num_tokens = length(all_tokens)

    aatype        = zeros(Int32, num_tokens)
    residue_index = zeros(Int32, num_tokens)
    token_index   = Int32.(1:num_tokens)
    asym_id       = zeros(Int32, num_tokens)
    entity_id     = zeros(Int32, num_tokens)
    sym_id        = ones(Int32, num_tokens)
    is_protein    = falses(num_tokens)
    is_rna        = falses(num_tokens)
    is_dna        = falses(num_tokens)
    is_ligand     = falses(num_tokens)
    token_mask    = trues(num_tokens)
    num_sym       = ones(Int32, num_tokens)

    # Assign chain-level metadata
    chain_to_asym  = Dict{String,Int32}()
    chain_to_entity = Dict{String,Int32}()
    asym_counter = Int32(0)
    entity_map = Dict{String,Int32}()  # sequence → entity_id
    entity_counter = Int32(0)

    for (ci, chain) in enumerate(fold_input.chains)
        cid = chain.id
        chain_to_asym[cid] = (asym_counter += 1)
        seq = chain.sequence
        if !haskey(entity_map, seq)
            entity_map[seq] = (entity_counter += 1)
        end
        chain_to_entity[cid] = entity_map[seq]
    end

    # Assign per-token features
    for (i, (cid, rid)) in enumerate(zip(all_tokens.chain_id, all_tokens.res_id))
        rn = all_tokens.res_name !== nothing ? all_tokens.res_name[i] : ""
        ct = all_tokens.chain_type !== nothing ? all_tokens.chain_type[i] : ""

        aatype[i] = Int32(get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, rn, 0))
        residue_index[i] = Int32(rid)
        asym_id[i]  = get(chain_to_asym,  cid, Int32(0))
        entity_id[i] = get(chain_to_entity, cid, Int32(0))

        is_protein[i] = (ct == PROTEIN_CHAIN)
        is_rna[i]     = (ct == RNA_CHAIN)
        is_dna[i]     = (ct == DNA_CHAIN)
        is_ligand[i]  = ct in LIGAND_CHAIN_TYPES
    end

    return Dict{String,Array}(
        "aatype"         => aatype,
        "residue_index"  => residue_index,
        "token_index"    => token_index,
        "asym_id"        => asym_id,
        "entity_id"      => entity_id,
        "sym_id"         => sym_id,
        "is_protein"     => is_protein,
        "is_rna"         => is_rna,
        "is_dna"         => is_dna,
        "is_ligand"      => is_ligand,
        "token_mask"     => token_mask,
        "num_sym"        => num_sym,
    )
end

"""
    compute_msa_features_for_chain(chain::Union{ProteinChain,RnaChain,DnaChain},
                                    num_tokens::Int,
                                    max_msa_seqs::Int=512) -> Dict{String,Array}
"""
function compute_msa_features_for_chain(
    chain,
    num_tokens::Int;
    max_msa_seqs::Int = 512,
)::Dict{String,Array}
    # Build MSA from chain's unpaired_msa
    msa_a3m = chain.unpaired_msa
    if msa_a3m === nothing || isempty(msa_a3m)
        msa_a3m = ">query\n$(chain.sequence)\n"
    end
    msa_obj = msa_from_a3m(chain.sequence, _chain_poly_type(chain), msa_a3m)
    msa_obj = truncate_msa(msa_obj, max_msa_seqs)

    msa_feats = featurize(msa_obj)  # from msa.jl
    n_seqs_actual = n_seqs(msa_obj)

    # Return with exact num_tokens columns
    function pad_to_tokens(arr, fill_val=0)
        cur_len = size(arr, 2)
        cur_len == num_tokens && return arr
        if cur_len < num_tokens
            pad = fill(eltype(arr)(fill_val), size(arr, 1), num_tokens - cur_len)
            return hcat(arr, pad)
        end
        return arr[:, 1:num_tokens]
    end

    msa_enc  = pad_to_tokens(get(msa_feats, "msa", zeros(Int8, 1, num_tokens)))
    msa_mask = pad_to_tokens(get(msa_feats, "msa_mask", ones(Bool, 1, num_tokens)))
    del_mat  = pad_to_tokens(get(msa_feats, "deletion_matrix", zeros(Int8, 1, num_tokens)))

    return Dict{String,Array}(
        "msa"              => msa_enc,
        "deletion_matrix"  => del_mat,
        "msa_mask"         => msa_mask,
        "cluster_bias_mask" => vcat([true], fill(false, n_seqs_actual - 1)),
    )
end

function _chain_poly_type(chain::ProteinChain)::String = PROTEIN_CHAIN
function _chain_poly_type(chain::RnaChain)::String     = RNA_CHAIN
function _chain_poly_type(chain::DnaChain)::String     = DNA_CHAIN
function _chain_poly_type(chain::Ligand)::String        = LIGAND_CHAIN

"""
    compute_ref_structure(all_tokens::AtomLayout, all_token_atoms::AtomLayout,
                           ccd::Ccd) -> Dict{String,Array}

Compute reference (ideal) atom positions for each token.
"""
function compute_ref_structure(
    all_tokens::AtomLayout,
    all_token_atoms::AtomLayout,
    ccd::Ccd,
)::Dict{String,Array}
    num_tokens = length(all_tokens)
    max_atoms = size(all_token_atoms.atom_name, 2)

    ref_pos      = zeros(Float32, num_tokens, max_atoms, 3)
    ref_mask     = falses(num_tokens, max_atoms)
    ref_element  = zeros(Int32, num_tokens, max_atoms)
    ref_charge   = zeros(Float32, num_tokens, max_atoms)
    ref_atom_chars = zeros(Int32, num_tokens, max_atoms, 4)
    ref_space_uid  = zeros(Int32, num_tokens, max_atoms)

    for i in 1:num_tokens
        rn = all_tokens.res_name !== nothing ? all_tokens.res_name[i] : ""
        comp = get(ccd, rn, nothing)
        comp === nothing && continue

        # Get ideal positions
        ideal_pos = get_ideal_positions(comp)  # Dict{atom_name => [x,y,z]}

        for k in 1:max_atoms
            an = all_token_atoms.atom_name[i, k]
            isempty(an) && continue

            pos = get(ideal_pos, an, nothing)
            if pos !== nothing
                ref_pos[i, k, :] = Float32.(pos)
                ref_mask[i, k]   = true
            end

            # Element index
            el = all_token_atoms.atom_element !== nothing ? all_token_atoms.atom_element[i, k] : ""
            ref_element[i, k] = Int32(get_element_index(el))

            # Atom name as 4 chars
            chars = atom_name_to_chars(an)
            ref_atom_chars[i, k, :] = chars
        end

        # Space UID: same residue = same UID
        ref_space_uid[i, :] .= Int32(i)
    end

    return Dict{String,Array}(
        "ref_pos"            => ref_pos,
        "ref_mask"           => ref_mask,
        "ref_element"        => ref_element,
        "ref_charge"         => ref_charge,
        "ref_atom_name_chars" => ref_atom_chars,
        "ref_space_uid"      => ref_space_uid,
    )
end
