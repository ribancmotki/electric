#!/usr/bin/env julia
"""
run_prediction.jl — CLI entry point for biomolecular structure prediction.

Usage:
  julia run_prediction.jl --json_path <input.json> --output_dir <output/> [options]
"""

using Pkg
Pkg.activate(dirname(@__FILE__))

using ArgParse
using Logging
using Dates
using Printf

include(joinpath(dirname(@__FILE__), "src", "BioStructurePrediction", "__init__.jl"))
using .BioStructurePrediction

# ──────────────────────────────────────────────
#  Argument parsing
# ──────────────────────────────────────────────

function parse_commandline()
    s = ArgParseSettings(
        prog        = "run_prediction.jl",
        description = "AlphaFold 3-style biomolecular structure prediction",
    )

    @add_arg_table! s begin
        "--json_path"
            help     = "Path to input JSON file or directory of JSON files"
            arg_type = String
            required = true
        "--output_dir"
            help     = "Directory for output files"
            arg_type = String
            required = true
        "--model_dir"
            help     = "Directory containing model parameters (*.h5 or *.npz)"
            arg_type = String
            default  = ENV["MODEL_PARAMETERS_PATH"] |> x -> isempty(x) ? "models" : x
        "--db_dir"
            help     = "Root directory for sequence and template databases"
            arg_type = String
            default  = get(ENV, "DB_DIR", "databases")
        "--num_diffusion_samples"
            help     = "Number of diffusion samples per seed"
            arg_type = Int
            default  = 5
        "--num_recycles"
            help     = "Number of Evoformer recycle iterations"
            arg_type = Int
            default  = 10
        "--seeds"
            help     = "Comma-separated list of RNG seeds (overrides seeds in JSON)"
            arg_type = String
            default  = ""
        "--buckets"
            help     = "Comma-separated token-count bucket sizes (e.g. 256,512,1024)"
            arg_type = String
            default  = "256,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120"
        "--conformer_max_iterations"
            help     = "Maximum RDKit conformer generation iterations (default: auto)"
            arg_type = Int
            default  = 0
        "--run_data_pipeline"
            help     = "Run the MSA/template search data pipeline"
            action   = :store_true
        "--jackhmmer_binary_path"
            help     = "Path to jackhmmer binary"
            arg_type = String
            default  = get(ENV, "JACKHMMER_BINARY_PATH", "")
        "--nhmmer_binary_path"
            help     = "Path to nhmmer binary"
            arg_type = String
            default  = get(ENV, "NHMMER_BINARY_PATH", "")
        "--hmmbuild_binary_path"
            help     = "Path to hmmbuild binary"
            arg_type = String
            default  = get(ENV, "HMMBUILD_BINARY_PATH", "")
        "--hmmsearch_binary_path"
            help     = "Path to hmmsearch binary"
            arg_type = String
            default  = get(ENV, "HMMSEARCH_BINARY_PATH", "")
        "--hmmalign_binary_path"
            help     = "Path to hmmalign binary"
            arg_type = String
            default  = get(ENV, "HMMALIGN_BINARY_PATH", "")
        "--flash_attention_implementation"
            help     = "Flash attention backend: triton, cudnn, or xla"
            arg_type = String
            default  = "xla"
        "--return_embeddings"
            help     = "Save single and pair embeddings to output"
            action   = :store_true
        "--return_distogram"
            help     = "Save distogram to output"
            action   = :store_true
        "--compress_large_output_files"
            help     = "Compress large output files with zstd"
            action   = :store_true
        "--use_gpu"
            help     = "Use GPU if available"
            action   = :store_true
        "--norun_data_pipeline"
            help     = "Skip data pipeline even if MSAs are absent"
            action   = :store_true
        "--max_template_date"
            help     = "Maximum template release date (YYYY-MM-DD)"
            arg_type = String
            default  = Dates.format(today() - Day(180), "yyyy-mm-dd")
        "--verbose"
            help     = "Print detailed logging"
            action   = :store_true
    end

    return parse_args(s)
end

# ──────────────────────────────────────────────
#  Helper: parse bucket list
# ──────────────────────────────────────────────

function parse_buckets(s::String)::Vector{Int}
    parts = split(s, ',')
    buckets = Int[]
    for p in parts
        p_trimmed = strip(p)
        isempty(p_trimmed) && continue
        push!(buckets, parse(Int, p_trimmed))
    end
    return sort(unique(buckets))
end

# ──────────────────────────────────────────────
#  Structure prediction entry point
# ──────────────────────────────────────────────

"""
    predict_structure(
        runner::ModelRunner,
        fold_input::FoldingInput,
        ccd::Ccd;
        buckets, compress, return_embeddings, return_distogram
    ) -> ResultsForSeed

Predict structure for a single FoldingInput (all seeds).
Returns ResultsForSeed with inference results, embeddings, and distogram.
"""
function predict_structure(
    runner::ModelRunner,
    fold_input::FoldingInput,
    ccd::Ccd;
    buckets::Vector{Int},
    compress::Bool,
    return_embeddings::Bool,
    return_distogram::Bool,
)::Vector{ResultsForSeed}
    results = ResultsForSeed[]

    for seed in fold_input.rng_seeds
        @info "Featurising for seed=$seed ..."
        seed_input = FoldingInput(
            fold_input.name,
            [seed],
            fold_input.protein_chains,
            fold_input.rna_chains,
            fold_input.dna_chains,
            fold_input.ligands,
            fold_input.bonded_atom_pairs,
            fold_input.user_ccd,
            fold_input.dialect,
            fold_input.version,
        )

        batch_dicts = featurise_input(seed_input; buckets=buckets, ccd=ccd)
        isempty(batch_dicts) && continue
        batch = first(batch_dicts)

        @info "Running model for seed=$seed ..."
        raw_result = predict(runner, batch, seed)

        n_tokens     = Int(batch["num_tokens"][1])
        token_chain_ids = batch["token_chain_ids"]

        # Extract per-sample structures and confidence metrics
        n_samples = runner.config.num_diffusion_samples
        inference_results = InferenceResult[]

        for si in 1:n_samples
            # Extract predicted positions for this sample
            sample_positions = raw_result["predicted_positions"][si, :, :, :]  # (n_tokens, n_slots, 3)
            token_residue_types = String.(batch["token_chain_ids"])  # placeholder; replace in full impl

            # Build Structure from predicted positions
            atom_mask = Float32.(batch["atom_mask"])
            pred_struct = build_structure_from_dense_positions(;
                positions           = sample_positions,
                mask                = atom_mask .> 0.5f0,
                token_residue_types = fill("ALA", n_tokens),
                token_chain_ids     = token_chain_ids,
                token_seq_ids       = string.(1:n_tokens),
                bfactors            = zeros(Float32, n_tokens),
                name                = fold_input.name,
            )

            # Confidence metrics
            conf = compute_confidence_metrics(raw_result, pred_struct, token_chain_ids)

            @info "Sample $si — ranking score: $(round(conf.ranking_score; digits=4))"

            meta = Dict{String,Any}("seed" => seed, "sample" => si)
            push!(inference_results, InferenceResult(pred_struct, conf, meta))
        end

        emb = if return_embeddings
            Dict{String,Array}(
                "single_embeddings" => raw_result["single_embeddings"],
                "pair_embeddings"   => raw_result["pair_embeddings"],
            )
        else
            nothing
        end

        dgram = return_distogram ? raw_result["distogram"]["distogram"] : nothing

        push!(results, ResultsForSeed(seed, inference_results, seed_input, emb, dgram))
    end

    return results
end

# ──────────────────────────────────────────────
#  Data pipeline runner
# ──────────────────────────────────────────────

"""
    run_data_pipeline(
        fold_input::FoldingInput,
        pipeline_config::DataPipelineConfig
    ) -> FoldingInput

Run the MSA / template search data pipeline on fold_input.
"""
function run_data_pipeline(
    fold_input::FoldingInput,
    pipeline_config::DataPipelineConfig,
)::FoldingInput
    dp = DataPipeline(pipeline_config)
    return process(dp, fold_input)
end

# ──────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────

function main()
    args = parse_commandline()

    # Logging level
    if args["verbose"]
        global_logger(ConsoleLogger(stderr, Logging.Debug))
    else
        global_logger(ConsoleLogger(stderr, Logging.Info))
    end

    @info "BioStructurePrediction v$(VERSION_STRING)"
    @info "Julia $(VERSION)"

    # Parse buckets
    buckets = parse_buckets(args["buckets"])
    @info "Token buckets: $buckets"

    # Load CCD
    ccd_path = get_ccd_database_path()
    if !isfile(ccd_path)
        @warn "CCD database not found at $ccd_path; ligand features will be degraded"
        ccd = Ccd(Dict())
    else
        @info "Loading CCD from $ccd_path ..."
        ccd = load_ccd(ccd_path)
    end

    # Model runner
    model_dir = args["model_dir"]
    if isdir(model_dir)
        @info "Loading model from $model_dir ..."
    else
        @warn "Model directory not found: $model_dir — predictions will use zero weights"
    end

    cfg = make_model_config(
        flash_attention_implementation = args["flash_attention_implementation"],
        num_diffusion_samples          = args["num_diffusion_samples"],
        num_recycles                   = args["num_recycles"],
        return_embeddings              = args["return_embeddings"],
        return_distogram               = args["return_distogram"],
    )

    runner = if isdir(model_dir)
        ModelRunner(model_dir; config=cfg, use_gpu=args["use_gpu"])
    else
        # Fallback: empty parameter dict
        ModelRunner(Dict{String,Array}(), cfg, false)
    end

    # Load fold inputs
    json_path = args["json_path"]
    if isdir(json_path)
        fold_inputs = load_fold_inputs_from_dir(json_path)
    else
        fold_inputs = load_fold_inputs_from_path(json_path)
    end
    @info "Loaded $(length(fold_inputs)) fold input(s)"

    # Override seeds if requested
    seed_override = if !isempty(args["seeds"])
        [parse(Int, strip(s)) for s in split(args["seeds"], ',')]
    else
        nothing
    end

    # Max template date
    max_template_date = Date(args["max_template_date"])

    # Data pipeline config
    pipeline_cfg = DataPipelineConfig(
        args["jackhmmer_binary_path"],
        args["nhmmer_binary_path"],
        args["hmmalign_binary_path"],
        args["hmmsearch_binary_path"],
        args["hmmbuild_binary_path"],
        # Databases (resolved from db_dir)
        joinpath(args["db_dir"], "small_bfd", "bfd-first_non_consensus_sequences.fasta"),
        joinpath(args["db_dir"], "mgnify", "mgy_clusters_2022_05.fa"),
        joinpath(args["db_dir"], "uniprot", "uniprot_all_2021_04.fa"),
        joinpath(args["db_dir"], "uniref90", "uniref90_2022_05.fa"),
        joinpath(args["db_dir"], "nt_rna", "nt_all_2023_02_23.fasta"),
        joinpath(args["db_dir"], "rfam", "rfam_14_4_clustered_rep_seq.fasta"),
        joinpath(args["db_dir"], "rnacentral", "rnacentral_active_seq_id_90_cov_80_linclust.fasta"),
        joinpath(args["db_dir"], "pdb", "pdb_2022_09_28_mmcifs.tar"),
        joinpath(args["db_dir"], "seqres", "pdb_seqres_2022_09_28.fasta"),
        # Z-values
        nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        # Parallelism
        8, nothing, 8, nothing,
        # Pipeline options
        max_template_date,
        true,
        args["conformer_max_iterations"] > 0 ? args["conformer_max_iterations"] : nothing,
    )

    # Process each fold input
    output_dir = args["output_dir"]
    mkpath(output_dir)

    for fold_input in fold_inputs
        name = sanitised_name(fold_input)
        @info "Processing: $name"

        # Apply seed override
        fi = if seed_override !== nothing
            FoldingInput(
                fold_input.name, seed_override,
                fold_input.protein_chains, fold_input.rna_chains, fold_input.dna_chains,
                fold_input.ligands, fold_input.bonded_atom_pairs,
                fold_input.user_ccd, fold_input.dialect, fold_input.version,
            )
        else
            fold_input
        end

        # Run data pipeline
        if args["run_data_pipeline"] && !args["norun_data_pipeline"]
            @info "Running data pipeline for $name ..."
            fi = run_data_pipeline(fi, pipeline_cfg)
        end

        # Predict
        @info "Running structure prediction for $name ..."
        all_results = predict_structure(runner, fi, ccd;
            buckets           = buckets,
            compress          = args["compress_large_output_files"],
            return_embeddings = args["return_embeddings"],
            return_distogram  = args["return_distogram"],
        )

        # Write outputs
        job_output_dir = joinpath(output_dir, name)
        mkpath(job_output_dir)

        write_outputs(all_results, job_output_dir;
            job_name                   = name,
            compress_large_output_files = args["compress_large_output_files"],
        )

        @info "Finished $name → $job_output_dir"
    end

    @info "All jobs complete."
end

# Entry point
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
