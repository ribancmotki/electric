"""
StructurePrediction.jl — top-level module for biomolecular structure prediction.

Implements an AlphaFold 3-style pipeline for predicting 3D atomic coordinates
of protein, RNA, DNA, and small-molecule complexes from sequence and chemical inputs.
"""
module StructurePrediction

using Dates
using LinearAlgebra
using Statistics
using Logging
using JSON3
using Random

# ──────────────────────────────────────────────────────────────────────────────
# Load order: foundations → structures → data → geometry → model → training
# ──────────────────────────────────────────────────────────────────────────────

# Parsers
include("parsers/cif_dict.jl")
include("parsers/fasta_iterator.jl")
include("parsers/msa_conversion.jl")

# Constants (no dependencies)
include("constants/periodic_table.jl")
include("constants/residue_names.jl")
include("constants/mmcif_names.jl")
include("constants/atom_types.jl")
include("constants/side_chains.jl")
include("constants/chemical_component_sets.jl")
include("constants/chemical_components.jl")

# Constants converters
include("constants/converters/ccd_pickle_gen.jl")
include("constants/converters/chemical_component_sets_gen.jl")

# Common utilities
include("common/base_config.jl")
include("common/safe_pickle.jl")
include("common/resources.jl")
include("common/folding_input.jl")

# Structure layer
include("structure/table.jl")
include("structure/structure_tables.jl")
include("structure/structure.jl")
include("structure/mmcif.jl")
include("structure/bonds.jl")
include("structure/chemical_components.jl")
include("structure/bioassemblies.jl")
include("structure/parsing.jl")

# Geometry
include("geometry/struct_of_array.jl")
include("geometry/vector.jl")
include("geometry/rotation_matrix.jl")
include("geometry/rigid_matrix_vector.jl")
include("geometry/utils.jl")

# Data pipeline tools
include("data/tools/subprocess_utils.jl")
include("data/tools/shards.jl")
include("data/tools/hmmbuild.jl")
include("data/tools/hmmalign.jl")
include("data/tools/hmmsearch.jl")
include("data/tools/jackhmmer.jl")
include("data/tools/nhmmer.jl")
include("data/tools/msa_tool.jl")
include("data/tools/rdkit_utils.jl")

# Data pipeline
include("data/parsers.jl")
include("data/msa_config.jl")
include("data/msa.jl")
include("data/msa_identifiers.jl")
include("data/msa_features.jl")
include("data/structure_stores.jl")
include("data/template_realign.jl")
include("data/templates.jl")
include("data/featurisation.jl")
include("data/pipeline.jl")

# Model atom layout
include("model/atom_layout/atom_layout.jl")

# Model components
include("model/components/haiku_modules.jl")
include("model/components/mapping.jl")
include("model/components/utils.jl")

# Model core
include("model/data_constants.jl")
include("model/model_config.jl")
include("model/confidence_types.jl")
include("model/params.jl")
include("model/features.jl")
include("model/feat_batch.jl")
include("model/merging_features.jl")
include("model/mmcif_metadata.jl")
include("model/msa_pairing.jl")
include("model/protein_data_processing.jl")
include("model/data3.jl")

# Model pipeline
include("model/pipeline/inter_chain_bonds.jl")
include("model/pipeline/structure_cleaning.jl")
include("model/pipeline/pipeline.jl")

# Model network
include("model/network/noise_level_embeddings.jl")
include("model/network/featurization.jl")
include("model/network/modules.jl")
include("model/network/diffusion_transformer.jl")
include("model/network/template_modules.jl")
include("model/network/atom_cross_attention.jl")
include("model/network/evoformer.jl")
include("model/network/confidence_head.jl")
include("model/network/diffusion_head.jl")
include("model/network/distogram_head.jl")

# Model top-level
include("model/confidences.jl")
include("model/scoring/alignment.jl")
include("model/scoring/chirality.jl")
include("model/scoring/scoring.jl")
include("model/post_processing.jl")
include("model/model.jl")

# Training
include("training/data_loader.jl")
include("training/loss.jl")
include("training/train.jl")

# ──────────────────────────────────────────────────────────────────────────────
# Public API exports
# ──────────────────────────────────────────────────────────────────────────────

# Common
export Input, ProteinChain, RnaChain, DnaChain, Ligand, Template
export load_fold_inputs_from_path, load_fold_inputs_from_dir, with_multiple_seeds
export to_json, from_json

# Constants
export PROTEIN_TYPES, RNA_TYPES, DNA_TYPES, NUCLEIC_TYPES
export PROTEIN_TYPES_WITH_UNKNOWN, NUCLEIC_TYPES_WITH_2_UNKS
export letters_three_to_one
export CCD_NAME_TO_ONE_LETTER

# Structure
export Structure, Table
export from_mmcif, to_mmcif, chains, coords, filter_to_entity_type

# Geometry
export Vec3Array, Rot3Array, Rigid3Array

# Atom layout
export AtomLayout, Residues, GatherInfo
export make_flat_atom_layout, tokenizer, compute_gather_idxs, convert

# Data pipeline
export DataPipelineConfig, DataPipeline
export Msa, MsaConfig, Templates

# Model
export ModelRunner, predict_structure, run_inference
export InferenceResult, ProcessedInferenceResult
export post_process_inference_result, write_output, write_embeddings
export load_params

# Constants re-exports
export Ccd

end # module StructurePrediction
