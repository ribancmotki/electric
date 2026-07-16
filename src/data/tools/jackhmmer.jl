"""
jackhmmer.jl — Julia wrapper for jackhmmer MSA search.
"""

"""
    run_jackhmmer(config::JackhmmerConfig, query_fasta::String) -> String

Run jackhmmer to produce an A3M MSA. Handles sharded databases.
Returns an A3M string.
"""
function run_jackhmmer(config::JackhmmerConfig, query_fasta::String)::String
    db_path = config.database_config.path

    if is_sharded_path(db_path)
        shard_paths = get_sharded_paths(db_path)
        run_one = path -> _run_jackhmmer_single(config, query_fasta, path)
        results = run_parallel_shards(run_one, shard_paths;
                                      max_parallel=config.max_parallel_shards)
        return merge_jackhmmer_results(String[r for r in results])
    else
        return _run_jackhmmer_single(config, query_fasta, db_path)
    end
end

function _run_jackhmmer_single(config::JackhmmerConfig, query_fasta::String,
                                db_path::String)::String
    mktempdir_cleanup() do tmpdir
        query_path = joinpath(tmpdir, "query.fasta")
        output_sto = joinpath(tmpdir, "output.sto")
        write(query_path, query_fasta)

        cmd_parts = [
            config.binary_path,
            "--noali",
            "--F1", "0.0005",
            "--F2", "0.00005",
            "--F3", "0.0000005",
            "--incE", "0.0001",
            "-E", "0.0001",
            "--cpu", string(config.n_cpu),
            "-N", string(config.n_iter),
        ]

        if config.z_value !== nothing
            append!(cmd_parts, ["-Z", string(config.z_value)])
        end
        if config.dom_z_value !== nothing
            append!(cmd_parts, ["--domZ", string(config.dom_z_value)])
        end
        append!(cmd_parts, ["-A", output_sto, query_path, db_path])

        cmd = Cmd(cmd_parts)
        result = run_subprocess(cmd; check=false)
        if result.exit_code != 0
            @warn "jackhmmer exited with code $(result.exit_code) on $db_path"
            return ">query\n$(get_query_sequence(query_fasta))\n"
        end

        !isfile(output_sto) && return ">query\n$(get_query_sequence(query_fasta))\n"
        sto_text = read(output_sto, String)
        a3m = stockholm_to_a3m(sto_text)

        # Crop to max_sequences
        if config.max_sequences > 0
            records = a3m_to_fasta(a3m)
            if length(records) > config.max_sequences
                records = records[1:config.max_sequences]
                a3m = fasta_to_a3m(records)
            end
        end
        return a3m
    end
end
