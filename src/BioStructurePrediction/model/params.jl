"""
Model parameter loading and validation.
"""

using HDF5
using Logging

# ──────────────────────────────────────────────
#  Expected parameter shapes
# ──────────────────────────────────────────────

"""
All expected parameter names, dtypes, and shapes.
This list defines the contract for loading model weights.
Keys match the HDF5 flat key format.
"""
const EXPECTED_PARAMS = Dict{String,Tuple{Type,Tuple}}(
    # ─── Confidence head ───────────────────────────────────────────────────────
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention1/act_norm:offset" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention1/act_norm:scale" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention1/gating_query:weights" => (UInt16, (4, 128, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention1/k_projection:weights" => (UInt16, (4, 4, 32, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention1/output_projection:weights" => (UInt16, (4, 128, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention1/pair_bias_projection:weights" => (UInt16, (4, 128, 4)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention1/q_projection:weights" => (UInt16, (4, 4, 32, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention1/v_projection:weights" => (UInt16, (4, 128, 4, 32)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention2/act_norm:offset" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention2/act_norm:scale" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention2/gating_query:weights" => (UInt16, (4, 128, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention2/k_projection:weights" => (UInt16, (4, 4, 32, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention2/output_projection:weights" => (UInt16, (4, 128, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention2/pair_bias_projection:weights" => (UInt16, (4, 128, 4)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention2/q_projection:weights" => (UInt16, (4, 4, 32, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_attention2/v_projection:weights" => (UInt16, (4, 128, 4, 32)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_transition/input_layer_norm:offset" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_transition/input_layer_norm:scale" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_transition/transition1:weights" => (UInt16, (4, 128, 1024)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/pair_transition/transition2:weights" => (UInt16, (4, 512, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_attention_gating_query:weights" => (UInt16, (4, 384, 384)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_attention_k_projection:weights" => (UInt16, (4, 384, 16, 24)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_attention_layer_norm:offset" => (Float32, (4, 384)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_attention_layer_norm:scale" => (Float32, (4, 384)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_attention_q_projection:bias" => (UInt16, (4, 16, 24)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_attention_q_projection:weights" => (UInt16, (4, 384, 16, 24)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_attention_transition2:weights" => (UInt16, (4, 384, 384)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_attention_v_projection:weights" => (UInt16, (4, 384, 16, 24)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_pair_logits_norm:offset" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_pair_logits_norm:scale" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_pair_logits_projection:weights" => (UInt16, (4, 128, 16)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_transition/input_layer_norm:offset" => (Float32, (4, 384)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_transition/input_layer_norm:scale" => (Float32, (4, 384)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_transition/transition1:weights" => (UInt16, (4, 384, 3072)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/single_transition/transition2:weights" => (UInt16, (4, 1536, 384)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_incoming/center_norm:offset" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_incoming/center_norm:scale" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_incoming/gate:weights" => (UInt16, (4, 128, 256)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_incoming/gating_linear:weights" => (UInt16, (4, 128, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_incoming/left_norm_input:offset" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_incoming/left_norm_input:scale" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_incoming/output_projection:weights" => (UInt16, (4, 128, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_incoming/projection:weights" => (UInt16, (4, 128, 256)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_outgoing/center_norm:offset" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_outgoing/center_norm:scale" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_outgoing/gate:weights" => (UInt16, (4, 128, 256)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_outgoing/gating_linear:weights" => (UInt16, (4, 128, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_outgoing/left_norm_input:offset" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_outgoing/left_norm_input:scale" => (Float32, (4, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_outgoing/output_projection:weights" => (UInt16, (4, 128, 128)),
    "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/triangle_multiplication_outgoing/projection:weights" => (UInt16, (4, 128, 256)),
    "diffuser/confidence_head/~_embed_features/distogram_feat_project:weights" => (UInt16, (39, 128)),
    "diffuser/confidence_head/~_embed_features/left_target_feat_project:weights" => (UInt16, (447, 128)),
    "diffuser/confidence_head/~_embed_features/right_target_feat_project:weights" => (UInt16, (447, 128)),
    "diffuser/confidence_head/experimentally_resolved_ln:offset" => (Float32, (384,)),
    "diffuser/confidence_head/experimentally_resolved_ln:scale" => (Float32, (384,)),
    "diffuser/confidence_head/experimentally_resolved_logits:weights" => (Float32, (384, 24, 2)),
    "diffuser/confidence_head/left_half_distance_logits:weights" => (Float32, (128, 64)),
    "diffuser/confidence_head/logits_ln:offset" => (Float32, (128,)),
    "diffuser/confidence_head/logits_ln:scale" => (Float32, (128,)),
    "diffuser/confidence_head/pae_logits_ln:offset" => (Float32, (128,)),
    "diffuser/confidence_head/pae_logits_ln:scale" => (Float32, (128,)),
    "diffuser/confidence_head/pae_logits:weights" => (Float32, (128, 64)),
    "diffuser/confidence_head/plddt_logits_ln:offset" => (Float32, (384,)),
    "diffuser/confidence_head/plddt_logits_ln:scale" => (Float32, (384,)),
    "diffuser/confidence_head/plddt_logits:weights" => (Float32, (384, 24, 50)),
    # ─── Distogram head ───────────────────────────────────────────────────────
    "diffuser/distogram_head/half_logits:weights" => (Float32, (128, 64)),
    # ─── Evoformer conditioning ───────────────────────────────────────────────
    "diffuser/evoformer_conditioning_embed_pair_distances_1:weights" => (Float32, (1, 16)),
    "diffuser/evoformer_conditioning_embed_pair_distances:weights" => (Float32, (1, 16)),
    "diffuser/evoformer_conditioning_embed_pair_offsets_1:weights" => (Float32, (3, 16)),
    "diffuser/evoformer_conditioning_embed_pair_offsets_valid:weights" => (Float32, (1, 16)),
    "diffuser/evoformer_conditioning_embed_pair_offsets:weights" => (Float32, (3, 16)),
    "diffuser/evoformer_conditioning_embed_ref_atom_name:weights" => (Float32, (256, 128)),
    "diffuser/evoformer_conditioning_embed_ref_charge:weights" => (Float32, (1, 128)),
    "diffuser/evoformer_conditioning_embed_ref_element:weights" => (Float32, (128, 128)),
    "diffuser/evoformer_conditioning_embed_ref_mask:weights" => (Float32, (1, 128)),
    "diffuser/evoformer_conditioning_embed_ref_pos:weights" => (Float32, (3, 128)),
    "diffuser/evoformer_conditioning_pair_input_layer_norm:scale" => (Float32, (16,)),
    "diffuser/evoformer_conditioning_pair_logits_projection:weights" => (Float32, (16, 3, 4)),
    "diffuser/evoformer_conditioning_pair_mlp_1:weights" => (Float32, (16, 16)),
    "diffuser/evoformer_conditioning_pair_mlp_2:weights" => (Float32, (16, 16)),
    "diffuser/evoformer_conditioning_pair_mlp_3:weights" => (Float32, (16, 16)),
    "diffuser/evoformer_conditioning_project_atom_features_for_aggr:weights" => (Float32, (128, 384)),
    "diffuser/evoformer_conditioning_single_to_pair_cond_col_1:weights" => (Float32, (128, 16)),
    "diffuser/evoformer_conditioning_single_to_pair_cond_col:weights" => (Float32, (128, 16)),
    "diffuser/evoformer_conditioning_single_to_pair_cond_row_1:weights" => (Float32, (128, 16)),
    "diffuser/evoformer_conditioning_single_to_pair_cond_row:weights" => (Float32, (128, 16)),
    # ─── Evoformer trunk (48 layers) ─────────────────────────────────────────
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/act_norm:offset" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/act_norm:scale" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/gating_query:weights" => (UInt16, (48, 128, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/k_projection:weights" => (UInt16, (48, 4, 32, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/output_projection:weights" => (UInt16, (48, 128, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/pair_bias_projection:weights" => (UInt16, (48, 128, 4)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/q_projection:weights" => (UInt16, (48, 4, 32, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention1/v_projection:weights" => (UInt16, (48, 128, 4, 32)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/act_norm:offset" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/act_norm:scale" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/gating_query:weights" => (UInt16, (48, 128, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/k_projection:weights" => (UInt16, (48, 4, 32, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/output_projection:weights" => (UInt16, (48, 128, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/pair_bias_projection:weights" => (UInt16, (48, 128, 4)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/q_projection:weights" => (UInt16, (48, 4, 32, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_attention2/v_projection:weights" => (UInt16, (48, 128, 4, 32)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_transition/input_layer_norm:offset" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_transition/input_layer_norm:scale" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_transition/transition1:weights" => (UInt16, (48, 128, 1024)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/pair_transition/transition2:weights" => (UInt16, (48, 512, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_gating_query:weights" => (UInt16, (48, 384, 384)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_k_projection:weights" => (UInt16, (48, 384, 16, 24)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_layer_norm:offset" => (Float32, (48, 384)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_layer_norm:scale" => (Float32, (48, 384)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_q_projection:bias" => (UInt16, (48, 16, 24)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_q_projection:weights" => (UInt16, (48, 384, 16, 24)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_transition2:weights" => (UInt16, (48, 384, 384)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_attention_v_projection:weights" => (UInt16, (48, 384, 16, 24)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_pair_logits_norm:offset" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_pair_logits_norm:scale" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_pair_logits_projection:weights" => (UInt16, (48, 128, 16)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_transition/input_layer_norm:offset" => (Float32, (48, 384)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_transition/input_layer_norm:scale" => (Float32, (48, 384)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_transition/transition1:weights" => (UInt16, (48, 384, 3072)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/single_transition/transition2:weights" => (UInt16, (48, 1536, 384)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/center_norm:offset" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/center_norm:scale" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/gate:weights" => (UInt16, (48, 128, 256)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/gating_linear:weights" => (UInt16, (48, 128, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/left_norm_input:offset" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/left_norm_input:scale" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/output_projection:weights" => (UInt16, (48, 128, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_incoming/projection:weights" => (UInt16, (48, 128, 256)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/center_norm:offset" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/center_norm:scale" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/gate:weights" => (UInt16, (48, 128, 256)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/gating_linear:weights" => (UInt16, (48, 128, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/left_norm_input:offset" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/left_norm_input:scale" => (Float32, (48, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/output_projection:weights" => (UInt16, (48, 128, 128)),
    "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/triangle_multiplication_outgoing/projection:weights" => (UInt16, (48, 128, 256)),
    # ─── Evoformer input embeddings ───────────────────────────────────────────
    "diffuser/evoformer/~_relative_encoding/position_activations:weights" => (UInt16, (139, 128)),
    "diffuser/evoformer/bond_embedding:weights" => (UInt16, (1, 128)),
    "diffuser/evoformer/extra_msa_target_feat:weights" => (UInt16, (447, 64)),
    "diffuser/evoformer/left_single:weights" => (UInt16, (447, 128)),
    "diffuser/evoformer/msa_activations:weights" => (UInt16, (34, 64)),
    "diffuser/evoformer/prev_embedding_layer_norm:offset" => (Float32, (128,)),
    "diffuser/evoformer/prev_embedding_layer_norm:scale" => (Float32, (128,)),
    "diffuser/evoformer/prev_embedding:weights" => (UInt16, (128, 128)),
    "diffuser/evoformer/prev_single_embedding_layer_norm:offset" => (Float32, (384,)),
    "diffuser/evoformer/prev_single_embedding_layer_norm:scale" => (Float32, (384,)),
    "diffuser/evoformer/prev_single_embedding:weights" => (UInt16, (384, 384)),
    "diffuser/evoformer/right_single:weights" => (UInt16, (447, 128)),
    "diffuser/evoformer/single_activations:weights" => (UInt16, (447, 384)),
    # ─── Template embedding ───────────────────────────────────────────────────
    "diffuser/evoformer/template_embedding/output_linear:weights" => (UInt16, (64, 128)),
    "diffuser/evoformer/template_embedding/single_template_embedding/output_layer_norm:offset" => (Float32, (64,)),
    "diffuser/evoformer/template_embedding/single_template_embedding/output_layer_norm:scale" => (Float32, (64,)),
    "diffuser/evoformer/template_embedding/single_template_embedding/query_embedding_norm:offset" => (Float32, (128,)),
    "diffuser/evoformer/template_embedding/single_template_embedding/query_embedding_norm:scale" => (Float32, (128,)),
    "diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_0:weights" => (UInt16, (39, 64)),
    "diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_1:weights" => (UInt16, (64,)),
    "diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_2:weights" => (UInt16, (31, 64)),
    "diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_3:weights" => (UInt16, (31, 64)),
    "diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_4:weights" => (UInt16, (64,)),
    "diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_5:weights" => (UInt16, (64,)),
    "diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_6:weights" => (UInt16, (64,)),
    "diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_7:weights" => (UInt16, (64,)),
    "diffuser/evoformer/template_embedding/single_template_embedding/template_pair_embedding_8:weights" => (UInt16, (128, 64)),
)

# ──────────────────────────────────────────────
#  Parameter loading
# ──────────────────────────────────────────────

"""
    get_model_params(model_dir::String) -> Dict{String,Array}

Read model parameters from model_dir.

Supports:
- HDF5 files (*.h5, *.hdf5) with flat key format
- NPZ files (*.npz) with flat key format
"""
function get_model_params(model_dir::String)::Dict{String,Array}
    isdir(model_dir) || error("Model directory not found: $model_dir")

    # Find parameter files
    h5_files  = filter(f -> endswith(f, ".h5") || endswith(f, ".hdf5"),
                       readdir(model_dir; join=true))
    npz_files = filter(f -> endswith(f, ".npz"),
                       readdir(model_dir; join=true))

    params = Dict{String,Array}()
    identifier = UInt8[]

    if !isempty(h5_files)
        param_file = first(h5_files)
        @info "Loading model parameters from $param_file"
        params = load_params_hdf5(param_file)
    elseif !isempty(npz_files)
        param_file = first(npz_files)
        @info "Loading model parameters from $param_file"
        params = load_params_npz(param_file)
    else
        error("No parameter files (*.h5, *.hdf5, *.npz) found in $model_dir")
    end

    # Try loading identifier
    id_path = joinpath(model_dir, "__meta__", "__identifier__")
    if isfile(id_path)
        params["__identifier__"] = read(id_path)
    end

    # Validate
    validate_params(params)

    return params
end

"""
    load_params_hdf5(path::String) -> Dict{String,Array}

Load a flat-key HDF5 parameter file.
"""
function load_params_hdf5(path::String)::Dict{String,Array}
    params = Dict{String,Array}()
    h5open(path, "r") do f
        _collect_hdf5_keys!(params, f, "")
    end
    return params
end

function _collect_hdf5_keys!(d::Dict, node, prefix::String)
    for name in keys(node)
        child = node[name]
        full_key = isempty(prefix) ? name : "$prefix/$name"
        if child isa HDF5.Dataset
            arr = read(child)
            # Convert bfloat16 (stored as UInt16) to Float32 lazily
            d[full_key] = arr
        else
            _collect_hdf5_keys!(d, child, full_key)
        end
    end
end

"""
    load_params_npz(path::String) -> Dict{String,Array}

Load a flat-key NPZ parameter file.
"""
function load_params_npz(path::String)::Dict{String,Array}
    using NPZ
    npz_data = npzread(path)
    return Dict{String,Array}(k => Array(v) for (k,v) in npz_data)
end

"""
    validate_params(params::Dict{String,Array})

Check that all expected parameter keys are present. Warns (does not error)
for missing keys to allow partial parameter files during development.
"""
function validate_params(params::Dict{String,Array})
    missing_keys = String[]
    for key in keys(EXPECTED_PARAMS)
        if !haskey(params, key)
            push!(missing_keys, key)
        end
    end
    if !isempty(missing_keys)
        @warn "$(length(missing_keys)) expected parameter keys not found in model file. " *
              "First missing: $(first(missing_keys))"
    else
        @info "All $(length(EXPECTED_PARAMS)) expected parameter keys found."
    end
end

"""
    bfloat16_to_float32(arr::Array{UInt16}) -> Array{Float32}

Convert an array stored as raw bfloat16 bytes (UInt16) to Float32.
"""
function bfloat16_to_float32(arr::Array{UInt16})::Array{Float32}
    out = Array{Float32}(undef, size(arr))
    for i in eachindex(arr)
        # Bfloat16: upper 16 bits of Float32
        bits = UInt32(arr[i]) << 16
        out[i] = reinterpret(Float32, bits)
    end
    return out
end

"""
    get_param_f32(params::Dict{String,Array}, key::String) -> Array{Float32}

Retrieve a parameter as Float32, converting from bfloat16 if necessary.
"""
function get_param_f32(params::Dict{String,Array}, key::String)::Array{Float32}
    arr = get(params, key, nothing)
    arr === nothing && error("Parameter not found: $key")
    if eltype(arr) == UInt16
        return bfloat16_to_float32(arr)
    elseif eltype(arr) == Float32
        return arr
    else
        return Float32.(arr)
    end
end
