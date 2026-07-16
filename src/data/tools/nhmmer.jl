"""
nhmmer.jl — Julia wrapper for nhmmer RNA/DNA MSA search.
"""

"""
    run_nhmmer(config::NhmmerConfig, query_fasta::String) -> String

Run nhmmer then hmmalign to produce an A3M MSA. Handles sharded databases.
"""
function run_nhmmer(config::NhmmerConfig, query_fasta::String)::String
    db_path = config.database_config.path

    if is_sharded_path(db_path)
        shard_paths = get_sharded_paths(db_path)
        run_one = path -> _run_nhmmer_single(config, query_fasta, path)
        results = run_parallel_shards(run_one, shard_paths;
                                      max_parallel=config.max_parallel_shards)
        return merge_nhmmer_results(String[r for r in results])
    else
        return _run_nhmmer_single(config, query_fasta, db_path)
    end
end

function _run_nhmmer_single(config::NhmmerConfig, query_fasta::String,
                             db_path::String)::String
    mktempdir_cleanup() do tmpdir
        query_path  = joinpath(tmpdir, "query.fasta")
        tblout_path = joinpath(tmpdir, "hits.tblout")
        sto_path    = joinpath(tmpdir, "hits.sto")
        write(query_path, query_fasta)

        cmd_parts = [
            config.binary_path,
            "--noali",
            "--cpu", string(config.n_cpu),
            "-E", string(config.e_value),
            "--incE", string(config.e_value),
            "--tblout", tblout_path,
            "-A", sto_path,
        ]
        if config.z_value !== nothing
            append!(cmd_parts, ["-Z", string(config.z_value)])
        end
        if config.alphabet != "rna"
            append!(cmd_parts, ["--$(config.alphabet)"])
        else
            append!(cmd_parts, ["--rna"])
        end
        append!(cmd_parts, [query_path, db_path])

        cmd = Cmd(cmd_parts)
        result = run_subprocess(cmd; check=false)
        if result.exit_code != 0 || !isfile(sto_path)
            query_seq = get_query_sequence(query_fasta)
            return ">query\n$query_seq\n"
        end

        sto_text = read(sto_path, String)

        # Run hmmalign to produce A3M
        hmmbuild_cfg = HmmbuildConfig(binary_path=config.hmmbuild_binary_path, n_cpu=1)
        hmmalign_cfg = HmmalignConfig(binary_path=config.hmmalign_binary_path)

        # Build HMM from query
        hmm = try
            run_hmmbuild(hmmbuild_cfg, ">query\n$(get_query_sequence(query_fasta))\n")
        catch e
            @warn "hmmbuild failed: $e"
            return stockholm_to_a3m(sto_text)
        end

        # Align hits to HMM
        hit_seqs_fasta = stockholm_to_a3m(sto_text)
        a3m = try
            aligned = run_hmmalign(hmmalign_cfg, hmm, hit_seqs_fasta)
            stockholm_to_a3m(aligned)
        catch e
            @warn "hmmalign failed: $e"
            stockholm_to_a3m(sto_text)
        end

        # Crop to max_sequences
        if config.max_sequences > 0
            records = a3m_to_fasta(a3m)
            length(records) > config.max_sequences && (records = records[1:config.max_sequences])
            a3m = fasta_to_a3m(records)
        end
        return a3m
    end
end
