"""
pipeline.jl — Data pipeline configuration and orchestration.
"""

using Dates
using Logging

# ──────────────────────────────────────────────────────────────────────────────
# DataPipelineConfig
# ──────────────────────────────────────────────────────────────────────────────

"""
    DataPipelineConfig

Configuration for the full data pipeline (MSA search + template search).
"""
struct DataPipelineConfig
    # Binary paths
    jackhmmer_binary_path::String
    nhmmer_binary_path::String
    hmmalign_binary_path::String
    hmmsearch_binary_path::String
    hmmbuild_binary_path::String

    # Database paths
    small_bfd_database_path::String
    mgnify_database_path::String
    uniprot_cluster_annot_database_path::String
    uniref90_database_path::String
    ntrna_database_path::String
    rfam_database_path::String
    rna_central_database_path::String
    seqres_database_path::String
    pdb_database_path::String

    # Z-values for database sizes
    small_bfd_z_value::Union{Int,Nothing}
    mgnify_z_value::Union{Int,Nothing}
    uniprot_cluster_annot_z_value::Union{Int,Nothing}
    uniref90_z_value::Union{Int,Nothing}
    ntrna_z_value::Union{Float64,Nothing}
    rfam_z_value::Union{Float64,Nothing}
    rna_central_z_value::Union{Float64,Nothing}

    # Parallelism
    jackhmmer_n_cpu::Int
    jackhmmer_max_parallel_shards::Union{Int,Nothing}
    nhmmer_n_cpu::Int
    nhmmer_max_parallel_shards::Union{Int,Nothing}

    # Template filtering
    max_template_date::Date
end

function DataPipelineConfig(;
    jackhmmer_binary_path::String  = "jackhmmer",
    nhmmer_binary_path::String     = "nhmmer",
    hmmalign_binary_path::String   = "hmmalign",
    hmmsearch_binary_path::String  = "hmmsearch",
    hmmbuild_binary_path::String   = "hmmbuild",
    small_bfd_database_path::String            = "",
    mgnify_database_path::String               = "",
    uniprot_cluster_annot_database_path::String = "",
    uniref90_database_path::String             = "",
    ntrna_database_path::String                = "",
    rfam_database_path::String                 = "",
    rna_central_database_path::String          = "",
    seqres_database_path::String               = "",
    pdb_database_path::String                  = "",
    small_bfd_z_value::Union{Int,Nothing}      = nothing,
    mgnify_z_value::Union{Int,Nothing}         = nothing,
    uniprot_cluster_annot_z_value::Union{Int,Nothing} = nothing,
    uniref90_z_value::Union{Int,Nothing}       = nothing,
    ntrna_z_value::Union{Float64,Nothing}      = nothing,
    rfam_z_value::Union{Float64,Nothing}       = nothing,
    rna_central_z_value::Union{Float64,Nothing} = nothing,
    jackhmmer_n_cpu::Int           = 8,
    jackhmmer_max_parallel_shards::Union{Int,Nothing} = nothing,
    nhmmer_n_cpu::Int              = 8,
    nhmmer_max_parallel_shards::Union{Int,Nothing} = nothing,
    max_template_date::Date        = Date(2021, 9, 30),
)
    return DataPipelineConfig(
        jackhmmer_binary_path, nhmmer_binary_path, hmmalign_binary_path,
        hmmsearch_binary_path, hmmbuild_binary_path,
        small_bfd_database_path, mgnify_database_path,
        uniprot_cluster_annot_database_path, uniref90_database_path,
        ntrna_database_path, rfam_database_path, rna_central_database_path,
        seqres_database_path, pdb_database_path,
        small_bfd_z_value, mgnify_z_value, uniprot_cluster_annot_z_value,
        uniref90_z_value, ntrna_z_value, rfam_z_value, rna_central_z_value,
        jackhmmer_n_cpu, jackhmmer_max_parallel_shards,
        nhmmer_n_cpu, nhmmer_max_parallel_shards,
        max_template_date,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# DataPipeline
# ──────────────────────────────────────────────────────────────────────────────

"""
    DataPipeline

Orchestrates MSA and template searches for all chains in an Input.
"""
struct DataPipeline
    config::DataPipelineConfig
    structure_store::PdbStructureStore
    _msa_cache::Dict{String,String}      # sequence → A3M
    _template_cache::Dict{String,Any}    # sequence → Templates
end

function DataPipeline(config::DataPipelineConfig)
    pdb_store = PdbStructureStore(config.pdb_database_path)
    return DataPipeline(config, pdb_store, Dict{String,String}(), Dict{String,Any}())
end

"""
    process(pipeline::DataPipeline, fold_input::Input) -> Input

Run MSA search and template search for all chains that need it.
Returns a new Input with all MSA and template fields populated.
"""
function process(pipeline::DataPipeline, fold_input::Input)::Input
    new_chains = AnyChain[]

    for chain in fold_input.chains
        push!(new_chains, _process_chain(pipeline, chain))
    end

    return Input(
        name  = fold_input.name,
        chains = new_chains,
        rng_seeds = fold_input.rng_seeds,
        bonded_atom_pairs = fold_input.bonded_atom_pairs,
        user_ccd = fold_input.user_ccd,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Per-chain processing
# ──────────────────────────────────────────────────────────────────────────────

function _process_chain(pipeline::DataPipeline, chain::ProteinChain)::ProteinChain
    cfg = pipeline.config

    # If all fields are already set, skip
    all_set = (chain.unpaired_msa !== nothing &&
               chain.paired_msa   !== nothing &&
               chain.templates    !== nothing)
    all_set && return chain

    seq = chain.sequence

    # Get or compute unpaired MSA
    unpaired_msa = chain.unpaired_msa
    if unpaired_msa === nothing
        unpaired_msa = get!(pipeline._msa_cache, "protein_unpaired_$seq") do
            _search_protein_unpaired_msa(pipeline, seq)
        end
    end

    # Get or compute paired MSA
    paired_msa = chain.paired_msa
    if paired_msa === nothing
        paired_msa = get!(pipeline._msa_cache, "protein_paired_$seq") do
            _search_protein_paired_msa(pipeline, seq)
        end
    end

    # Get or compute templates
    templates = chain.templates
    if templates === nothing && !isempty(cfg.seqres_database_path)
        cached_templates = get!(pipeline._template_cache, "protein_templates_$seq") do
            _search_protein_templates(pipeline, seq, unpaired_msa)
        end
        templates = _templates_to_input(cached_templates)
    end
    templates === nothing && (templates = Template[])

    return ProteinChain(
        id        = chain.id,
        sequence  = chain.sequence,
        ptms      = chain.ptms,
        description = chain.description,
        paired_msa   = paired_msa,
        unpaired_msa = unpaired_msa,
        templates    = templates,
    )
end

function _process_chain(pipeline::DataPipeline, chain::RnaChain)::RnaChain
    cfg = pipeline.config
    chain.unpaired_msa !== nothing && return chain

    seq = chain.sequence
    unpaired_msa = get!(pipeline._msa_cache, "rna_$seq") do
        _search_rna_msa(pipeline, seq)
    end

    return RnaChain(
        id           = chain.id,
        sequence     = chain.sequence,
        modifications = chain.modifications,
        description  = chain.description,
        unpaired_msa = unpaired_msa,
    )
end

function _process_chain(pipeline::DataPipeline, chain::DnaChain)::DnaChain
    return chain  # DNA chains: no MSA search
end

function _process_chain(pipeline::DataPipeline, chain::Ligand)::Ligand
    return chain  # Ligands: no MSA search
end

# ──────────────────────────────────────────────────────────────────────────────
# Protein MSA search
# ──────────────────────────────────────────────────────────────────────────────

function _search_protein_unpaired_msa(pipeline::DataPipeline, seq::String)::String
    cfg = pipeline.config
    query = make_single_record_fasta(seq)
    msas = String[]

    tasks = Task[]
    dbs = [
        (cfg.uniref90_database_path,   cfg.uniref90_z_value,   10000),
        (cfg.mgnify_database_path,     cfg.mgnify_z_value,     5000),
        (cfg.small_bfd_database_path,  cfg.small_bfd_z_value,  5000),
    ]

    for (db_path, z_val, max_seqs) in dbs
        isempty(db_path) && continue
        t = Threads.@spawn begin
            jc = JackhmmerConfig(
                binary_path     = cfg.jackhmmer_binary_path,
                database_config = DatabaseConfig("db", db_path),
                n_cpu           = cfg.jackhmmer_n_cpu,
                n_iter          = 1,
                z_value         = z_val,
                max_sequences   = max_seqs,
                max_parallel_shards = cfg.jackhmmer_max_parallel_shards,
            )
            run_jackhmmer(jc, query)
        end
        push!(tasks, t)
    end

    for t in tasks
        a3m = fetch(t)
        !isempty(a3m) && push!(msas, a3m)
    end

    isempty(msas) && return ">query\n$seq\n"
    return merge_jackhmmer_results(msas)
end

function _search_protein_paired_msa(pipeline::DataPipeline, seq::String)::String
    cfg = pipeline.config
    isempty(cfg.uniprot_cluster_annot_database_path) && return ">query\n$seq\n"
    query = make_single_record_fasta(seq)
    jc = JackhmmerConfig(
        binary_path     = cfg.jackhmmer_binary_path,
        database_config = DatabaseConfig("uniprot", cfg.uniprot_cluster_annot_database_path),
        n_cpu           = cfg.jackhmmer_n_cpu,
        n_iter          = 1,
        z_value         = cfg.uniprot_cluster_annot_z_value,
        max_sequences   = 50000,
        max_parallel_shards = cfg.jackhmmer_max_parallel_shards,
    )
    return run_jackhmmer(jc, query)
end

function _search_protein_templates(pipeline::DataPipeline, seq::String,
                                    msa_a3m::String)::Templates
    cfg = pipeline.config
    hmssearch_cfg = HmmsearchConfig(
        hmmsearch_binary_path = cfg.hmmsearch_binary_path,
        hmmbuild_binary_path  = cfg.hmmbuild_binary_path,
        alphabet              = "amino",
    )
    filter_cfg = TemplateFilterConfig(max_template_date=cfg.max_template_date)
    return templates_from_seq_and_a3m(
        seq, msa_a3m, cfg.max_template_date,
        cfg.seqres_database_path, hmssearch_cfg, 1000,
        PROTEIN_CHAIN, pipeline.structure_store, filter_cfg,
    )
end

function _search_rna_msa(pipeline::DataPipeline, seq::String)::String
    cfg = pipeline.config
    query = make_single_record_fasta(seq)
    msas = String[]
    tasks = Task[]

    dbs = [
        (cfg.ntrna_database_path,       cfg.ntrna_z_value),
        (cfg.rfam_database_path,        cfg.rfam_z_value),
        (cfg.rna_central_database_path, cfg.rna_central_z_value),
    ]

    for (db_path, z_val) in dbs
        isempty(db_path) && continue
        t = Threads.@spawn begin
            nc = NhmmerConfig(
                binary_path            = cfg.nhmmer_binary_path,
                hmmalign_binary_path   = cfg.hmmalign_binary_path,
                hmmbuild_binary_path   = cfg.hmmbuild_binary_path,
                database_config        = DatabaseConfig("db", db_path),
                n_cpu                  = cfg.nhmmer_n_cpu,
                z_value                = z_val,
                max_sequences          = 10000,
                max_parallel_shards    = cfg.nhmmer_max_parallel_shards,
            )
            run_nhmmer(nc, query)
        end
        push!(tasks, t)
    end

    for t in tasks
        a3m = fetch(t)
        !isempty(a3m) && push!(msas, a3m)
    end

    isempty(msas) && return ">query\n$seq\n"
    return merge_nhmmer_results(msas)
end

function _templates_to_input(templates::Templates)::Vector{Template}
    result = Template[]
    for (hit, struc) in get_hits_with_structures(templates)
        struc === nothing && continue
        # Build a simple identity mapping
        q_len = length(templates.query_sequence)
        mapping = Dict{Int,Int}(i => i for i in 1:q_len)
        push!(result, Template(to_mmcif(struc), mapping))
    end
    return result
end
