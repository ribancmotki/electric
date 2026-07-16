"""
Data pipeline orchestration: MSA search, template search, conformer generation.
"""

using Dates
using Logging

# ──────────────────────────────────────────────
#  Utility
# ──────────────────────────────────────────────

"""
    _stockholm_to_a3m(sto_text::String) -> String

Convert a Stockholm-format alignment (from jackhmmer -A output) to A3M format.
"""
function _stockholm_to_a3m(sto_text::String)::String
    sequences = parse_stockholm(sto_text)
    isempty(sequences) && return ""
    buf = IOBuffer()
    for (name, seq) in sequences
        println(buf, ">$name")
        println(buf, seq)
    end
    return String(take!(buf))
end

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

"""
    DataPipelineConfig

All configuration required for running the data pipeline.
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
    pdb_database_path::String
    seqres_database_path::String

    # Z-values (optional)
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

    # Pipeline options
    max_template_date::Date
    resolve_msa_overlaps::Bool
    conformer_max_iterations::Union{Int,Nothing}
end

# ──────────────────────────────────────────────
#  DataPipeline
# ──────────────────────────────────────────────

"""
    DataPipeline

Executes the data preparation pipeline for biomolecular structure prediction.
"""
struct DataPipeline
    config::DataPipelineConfig
end

"""
    process(dp::DataPipeline, fold_input::FoldingInput) -> FoldingInput

Run the full data pipeline on a fold input and return the enriched FoldingInput
with MSAs, templates, and conformer coordinates filled in.
"""
function process(dp::DataPipeline, fold_input::FoldingInput)::FoldingInput
    cfg = dp.config

    new_protein_chains = ProteinChain[]
    new_rna_chains     = RnaChain[]
    new_dna_chains     = copy(fold_input.dna_chains)
    new_ligands        = copy(fold_input.ligands)

    # ── 6a: Protein MSA Search ─────────────────
    for chain in fold_input.protein_chains
        if chain.unpaired_msa !== nothing && chain.paired_msa !== nothing
            push!(new_protein_chains, chain)
            continue
        end

        query_seq = chain.sequence
        @info "Running MSA search for protein chain $(chain.ids)..."

        # UniRef90 (also used for template profile)
        uniref90_msa = run_jackhmmer_search(
            query_seq, cfg.uniref90_database_path,
            cfg.jackhmmer_binary_path, cfg.jackhmmer_n_cpu,
            cfg.uniref90_z_value, cfg.jackhmmer_max_parallel_shards,
        )

        # Small BFD
        bfd_msa = run_jackhmmer_search(
            query_seq, cfg.small_bfd_database_path,
            cfg.jackhmmer_binary_path, cfg.jackhmmer_n_cpu,
            cfg.small_bfd_z_value, cfg.jackhmmer_max_parallel_shards,
        )

        # MGnify
        mgnify_msa = run_jackhmmer_search(
            query_seq, cfg.mgnify_database_path,
            cfg.jackhmmer_binary_path, cfg.jackhmmer_n_cpu,
            cfg.mgnify_z_value, cfg.jackhmmer_max_parallel_shards,
        )

        # UniProt (paired MSA source)
        uniprot_msa = run_jackhmmer_search(
            query_seq, cfg.uniprot_cluster_annot_database_path,
            cfg.jackhmmer_binary_path, cfg.jackhmmer_n_cpu,
            cfg.uniprot_cluster_annot_z_value, cfg.jackhmmer_max_parallel_shards,
        )

        # Merge unpaired MSAs
        unpaired_merged = merge_msas([uniref90_msa, bfd_msa, mgnify_msa])

        # Resolve overlaps if requested
        if cfg.resolve_msa_overlaps
            unpaired_merged = deduplicate_unpaired_against_paired(unpaired_merged, uniprot_msa)
        end

        unpaired_a3m = msa_to_a3m(unpaired_merged)
        paired_a3m   = msa_to_a3m(uniprot_msa)

        # Template search
        templates = chain.templates
        if templates === nothing
            template_cfg = TemplateSearchConfig(
                4,
                cfg.max_template_date,
                cfg.pdb_database_path,
                cfg.seqres_database_path,
                cfg.hmmsearch_binary_path,
                cfg.hmmbuild_binary_path,
                cfg.hmmalign_binary_path,
            )
            hits = search_templates(query_seq, uniref90_msa, template_cfg)
            templates = template_hits_to_input(hits)
        end

        push!(new_protein_chains, ProteinChain(
            chain.ids, chain.sequence, chain.modifications,
            unpaired_a3m, paired_a3m, templates,
        ))
    end

    # ── 6b: RNA MSA Search ─────────────────────
    for chain in fold_input.rna_chains
        if chain.unpaired_msa !== nothing
            push!(new_rna_chains, chain)
            continue
        end

        query_seq = chain.sequence
        @info "Running MSA search for RNA chain $(chain.ids)..."

        ntrna_msa   = run_nhmmer_search(query_seq, cfg.ntrna_database_path,
                          cfg.nhmmer_binary_path, cfg.nhmmer_n_cpu,
                          cfg.ntrna_z_value, cfg.nhmmer_max_parallel_shards)
        rfam_msa    = run_nhmmer_search(query_seq, cfg.rfam_database_path,
                          cfg.nhmmer_binary_path, cfg.nhmmer_n_cpu,
                          cfg.rfam_z_value, cfg.nhmmer_max_parallel_shards)
        rnac_msa    = run_nhmmer_search(query_seq, cfg.rna_central_database_path,
                          cfg.nhmmer_binary_path, cfg.nhmmer_n_cpu,
                          cfg.rna_central_z_value, cfg.nhmmer_max_parallel_shards)

        unpaired_merged = merge_msas([ntrna_msa, rfam_msa, rnac_msa])
        unpaired_a3m    = msa_to_a3m(unpaired_merged)

        push!(new_rna_chains, RnaChain(
            chain.ids, chain.sequence, chain.modifications,
            unpaired_a3m, chain.paired_msa,
        ))
    end

    return FoldingInput(
        fold_input.name,
        fold_input.rng_seeds,
        new_protein_chains,
        new_rna_chains,
        new_dna_chains,
        new_ligands,
        fold_input.bonded_atom_pairs,
        fold_input.user_ccd,
        fold_input.dialect,
        fold_input.version,
    )
end

# ──────────────────────────────────────────────
#  Jackhmmer runner
# ──────────────────────────────────────────────

"""
    run_jackhmmer_search(
        query_seq, database_path, binary_path, n_cpu, z_value, max_parallel_shards
    ) -> Msa

Run Jackhmmer on a single database or sharded database.
"""
function run_jackhmmer_search(
    query_seq::String,
    database_path::String,
    binary_path::String,
    n_cpu::Int,
    z_value::Union{Int,Nothing},
    max_parallel_shards::Union{Int,Nothing},
)::Msa
    if isempty(binary_path) || !isfile(binary_path)
        @warn "jackhmmer binary not found: $binary_path; returning empty MSA"
        query_aln = replace(query_seq, r"[a-z]" => s -> uppercase(s))
        return Msa([query_aln], ["query"], zeros(Int, 1, length(query_aln)))
    end
    if !isfile(database_path) && !occursin(r"@\d+$|(-\d{5}-of-\d{5})$", database_path)
        @warn "Database not found: $database_path; returning empty MSA"
        query_aln = uppercase(query_seq)
        return Msa([query_aln], ["query"], zeros(Int, 1, length(query_aln)))
    end

    shard_paths = get_sharded_paths(database_path)
    if shard_paths !== nothing
        # Sharded search
        run_fn = path -> _run_jackhmmer_single(query_seq, path, binary_path, n_cpu, z_value)
        results = run_parallel_shards(shard_paths, run_fn, max_parallel_shards)
        merged_a3m = merge_jackhmmer_results(results)
        return Msa_from_a3m(merged_a3m)
    else
        a3m = _run_jackhmmer_single(query_seq, database_path, binary_path, n_cpu, z_value)
        return Msa_from_a3m(a3m)
    end
end

function _run_jackhmmer_single(
    query_seq::String,
    database_path::String,
    binary_path::String,
    n_cpu::Int,
    z_value::Union{Int,Nothing},
)::String
    tmp_dir    = mktempdir()
    query_file = joinpath(tmp_dir, "query.fasta")
    output_sto = joinpath(tmp_dir, "output.sto")
    try
        open(query_file, "w") do io
            println(io, ">query")
            println(io, query_seq)
        end

        cmd_parts = String[binary_path,
            "--noali",
            "--cpu", string(n_cpu),
            "-A", output_sto,
            "--tblout", joinpath(tmp_dir, "hits.tblout"),
        ]

        if z_value !== nothing
            push!(cmd_parts, "--Z", string(z_value))
            push!(cmd_parts, "--domZ", string(z_value))
        end
        push!(cmd_parts, query_file)
        push!(cmd_parts, database_path)

        cmd = Cmd(cmd_parts)
        run(pipeline(cmd; stdout=devnull, stderr=devnull))

        isfile(output_sto) || return ">query\n$query_seq\n"
        sto_text = read(output_sto, String)
        return _stockholm_to_a3m(sto_text)
    catch e
        @warn "jackhmmer failed for database $database_path: $e"
        return ">query\n$query_seq\n"
    finally
        rm(tmp_dir; recursive=true, force=true)
    end
end

# ──────────────────────────────────────────────
#  Nhmmer runner
# ──────────────────────────────────────────────

"""
    run_nhmmer_search(
        query_seq, database_path, binary_path, n_cpu, z_value, max_parallel_shards
    ) -> Msa

Run Nhmmer on a single database or sharded database.
"""
function run_nhmmer_search(
    query_seq::String,
    database_path::String,
    binary_path::String,
    n_cpu::Int,
    z_value::Union{Float64,Nothing},
    max_parallel_shards::Union{Int,Nothing},
)::Msa
    if isempty(binary_path) || !isfile(binary_path)
        @warn "nhmmer binary not found: $binary_path; returning empty MSA"
        return Msa([query_seq], ["query"], zeros(Int, 1, length(query_seq)))
    end
    if !isfile(database_path) && !occursin(r"@\d+$|(-\d{5}-of-\d{5})$", database_path)
        @warn "Database not found: $database_path; returning empty MSA"
        return Msa([query_seq], ["query"], zeros(Int, 1, length(query_seq)))
    end

    shard_paths = get_sharded_paths(database_path)
    if shard_paths !== nothing
        run_fn = path -> _run_nhmmer_single(query_seq, path, binary_path, n_cpu, z_value)
        results = run_parallel_shards(shard_paths, run_fn, max_parallel_shards)
        merged_sto = merge_nhmmer_results(results)
        return Msa_from_stockholm(merged_sto)
    else
        sto = _run_nhmmer_single(query_seq, database_path, binary_path, n_cpu, z_value)
        return Msa_from_stockholm(sto)
    end
end

function _run_nhmmer_single(
    query_seq::String,
    database_path::String,
    binary_path::String,
    n_cpu::Int,
    z_value::Union{Float64,Nothing},
)::String
    tmp_dir    = mktempdir()
    query_file = joinpath(tmp_dir, "query.fasta")
    output_sto = joinpath(tmp_dir, "output.sto")
    try
        open(query_file, "w") do io
            println(io, ">query")
            println(io, query_seq)
        end

        cmd_parts = String[binary_path,
            "--noali",
            "--rna",
            "--cpu", string(n_cpu),
            "-A", output_sto,
        ]

        if z_value !== nothing
            # z_value is in megabases for nhmmer
            push!(cmd_parts, "--Z", string(Int(round(z_value * 1e6))))
        end
        push!(cmd_parts, database_path)
        push!(cmd_parts, query_file)

        cmd = Cmd(cmd_parts)
        run(pipeline(cmd; stdout=devnull, stderr=devnull))

        isfile(output_sto) || return "# STOCKHOLM 1.0\nquery\t$query_seq\n//"
        return read(output_sto, String)
    catch e
        @warn "nhmmer failed for database $database_path: $e"
        return "# STOCKHOLM 1.0\nquery\t$query_seq\n//"
    finally
        rm(tmp_dir; recursive=true, force=true)
    end
end
