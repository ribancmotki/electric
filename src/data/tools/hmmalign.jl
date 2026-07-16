"""
hmmalign.jl — Julia wrapper for the hmmalign binary.
"""

"""
HmmalignConfig: configuration for hmmalign.
"""
struct HmmalignConfig
    binary_path::String
end

HmmalignConfig(; binary_path::String="hmmalign") = HmmalignConfig(binary_path)

"""
    run_hmmalign(config::HmmalignConfig, hmm_profile::String,
                 sequences_fasta::String; trim::Bool=true) -> String

Align sequences to an HMM profile using hmmalign.
Returns a Stockholm format alignment.
"""
function run_hmmalign(config::HmmalignConfig, hmm_profile::String,
                      sequences_fasta::String; trim::Bool=true)::String
    mktempdir_cleanup() do tmpdir
        hmm_path = joinpath(tmpdir, "query.hmm")
        seq_path = joinpath(tmpdir, "sequences.fasta")
        out_path = joinpath(tmpdir, "aligned.sto")
        write(hmm_path, hmm_profile)
        write(seq_path, sequences_fasta)

        trim_flag = trim ? `--trim` : ``
        cmd = `$(config.binary_path)
               $trim_flag
               --outformat afa
               -o $(out_path)
               $(hmm_path)
               $(seq_path)`

        run_subprocess(cmd)
        isfile(out_path) || error("hmmalign did not produce output at $out_path")
        return read(out_path, String)
    end
end
