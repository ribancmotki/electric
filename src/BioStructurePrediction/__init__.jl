"""
BioStructurePrediction module root.
Includes all submodules in dependency order.
"""

module BioStructurePrediction

using Reexport

# ──────────────────────────────────────────────
#  Standard library deps
# ──────────────────────────────────────────────
using Base: @kwdef
using Dates
using Logging
using Printf
using Statistics
using Random
using LinearAlgebra

# ──────────────────────────────────────────────
#  External package deps (loaded once here)
# ──────────────────────────────────────────────
using JSON3
using HDF5
using CodecZstd
using CSV
using DataFrames
using NPZ

# Optional GPU
try
    using CUDA
catch
    @info "CUDA not available — running on CPU"
end

# ──────────────────────────────────────────────
#  1. Version
# ──────────────────────────────────────────────
include("version.jl")

# ──────────────────────────────────────────────
#  2. Constants
# ──────────────────────────────────────────────
include("constants/periodic_table.jl")
include("constants/atom_types.jl")
include("constants/residue_names.jl")
include("constants/mmcif_names.jl")
include("constants/side_chains.jl")
include("constants/chemical_component_sets.jl")
include("constants/chemical_components.jl")

# ──────────────────────────────────────────────
#  3. Common utilities
# ──────────────────────────────────────────────
include("common/base_config.jl")
include("common/safe_pickle.jl")
include("common/resources.jl")

# ──────────────────────────────────────────────
#  4. Structure module
# ──────────────────────────────────────────────
include("structure/table.jl")
include("structure/structure.jl")
include("structure/mmcif.jl")
include("structure/bonds.jl")
include("structure/bioassemblies.jl")
include("structure/chemical_components.jl")
include("structure/parsing.jl")
include("structure/test_utils.jl")

# ──────────────────────────────────────────────
#  5. Data module
# ──────────────────────────────────────────────
include("data/parsers.jl")
include("data/msa.jl")
include("data/msa_config.jl")
include("data/msa_identifiers.jl")
include("data/msa_features.jl")
include("data/msa_pairing.jl")
include("data/tools/shards.jl")
include("data/structure_stores.jl")
include("data/template_realign.jl")
include("data/templates.jl")
include("common/folding_input.jl")
include("data/pipeline.jl")
include("data/featurisation.jl")

# ──────────────────────────────────────────────
#  6. Model module (geometry first, then constants, then components)
# ──────────────────────────────────────────────
include("jax/geometry/geometry.jl")
include("model/data_constants.jl")
include("model/model_config.jl")
include("model/confidence_types.jl")
include("model/params.jl")
include("model/components/utils.jl")
include("model/atom_layout/atom_layout.jl")
include("model/features.jl")
include("model/feat_batch.jl")
include("model/merging_features.jl")
include("model/mmcif_metadata.jl")
include("model/msa_pairing.jl")
include("model/protein_data_processing.jl")
include("model/confidences.jl")
include("model/data3.jl")
include("model/scoring/alignment.jl")
include("model/post_processing.jl")
include("model/model.jl")

# ──────────────────────────────────────────────
#  7. Testing utilities
# ──────────────────────────────────────────────
include("common/testing/data.jl")

# ──────────────────────────────────────────────
#  Exports (public API)
# ──────────────────────────────────────────────

# Version
export VERSION_STRING

# Folding input types
export ProteinChain, RnaChain, DnaChain, LigandEntity, BondedAtomPair, FoldingInput
export FoldingInput_from_json, FoldingInput_to_json
export load_fold_inputs_from_path, load_fold_inputs_from_dir, write_fold_input_json
export with_multiple_seeds, sanitised_name

# Chemical components
export Ccd, CcdAtom, CcdBond, CcdComponent
export load_ccd, get_component, get_component_atoms, get_ccd_database_path

# Structure types
export Structure, ChainInfo, ResidueInfo, Bond, StructureTable
export num_atoms, atom_positions, get_chain, get_residue
export structure_from_arrays, to_mmcif, from_mmcif
export parse_structure_from_mmcif_string, parse_structure_from_mmcif_file

# MSA types and functions
export Msa, Msa_from_a3m, Msa_from_stockholm, Msa_from_fasta
export msa_to_a3m, truncate_msa, merge_msas, deduplicate_unpaired_against_paired
export make_msa_features, make_extra_msa_features
export MsaConfig, default_msa_config

# Template search
export TemplateSearchConfig, TemplateHit, search_templates, template_hits_to_input

# Data pipeline
export DataPipelineConfig, DataPipeline
export process

# Featurisation
export featurise_input, compute_num_tokens
export BatchDict, assemble_batch, pad_to_bucket, select_bucket

# Model
export ModelConfig, make_model_config
export ModelRunner, predict
export ModelResult

# Confidence
export ConfidenceMetrics, SummaryConfidences, InferenceResult, ResultsForSeed
export compute_confidence_metrics, compute_plddt, compute_pae, compute_ptm
export detect_clashes, compute_ranking_score

# Post-processing / output
export write_output, write_outputs, write_embeddings, write_distogram

# Alignment
export compute_rmsd, superimpose, compute_tm_score

# Geometry
export Rigid3Array, compose, apply, invert, make_backbone_frames
export gram_schmidt_qr, pairwise_distances, dihedral_angle

# Parameters
export get_model_params, get_param_f32, bfloat16_to_float32, validate_params

# Data constants
export RESTYPE_ORDER, NUM_RESIDUE_TYPES, restype_index
export PLDDT_NUM_BINS, PAE_NUM_BINS, DISTOGRAM_NUM_BINS
export PLDDT_BIN_CENTERS, PAE_BIN_CENTERS, DISTOGRAM_BIN_CENTERS
export TARGET_FEAT_DIM, MSA_FEAT_DIM, PAIR_FEAT_DIM, SINGLE_FEAT_DIM

# Atom layout
export atom_layout_to_flat, flat_to_atom_layout, get_token_atom_mask
export dense_positions_from_structure, build_structure_from_dense_positions

# Resource utilities
export get_package_data_dir, replace_db_dir

# Testing helpers
export get_test_fold_input_5tgy

end  # module BioStructurePrediction
