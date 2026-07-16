"""
hmmbuild.jl — Julia wrapper for the hmmbuild binary.
"""

"""
HmbuildConfig: configuration for hmmbuild.
"""
struct HmmbuildConfig
    binary_path::String
    n_cpu::Int
end

HmmbuildConfig(; binary_path::String="hmmbuild", n_cpu::Int=4) =
    HmmbuildConfig(binary_path, n_cpu)

"""
    run_hmmbuild(config::HmmbuildConfig, msa_a3m::String, hmm_name::String="query")
        -> String

Run hmmbuild on an A3M MSA, return the HMM profile as a string.
"""
function run_hmmbuild(config::HmmbuildConfig, msa_a3m::String,
                      hmm_name::String="query")::String
    mktempdir_cleanup() do tmpdir
        msa_path = joinpath(tmpdir, "input.a3m")
        hmm_path = joinpath(tmpdir, "output.hmm")
        write(msa_path, msa_a3m)

        cmd = `$(config.binary_path)
               --cpu $(config.n_cpu)
               --hand
               --symfrac 0
               --fragthresh 0
               -o /dev/null
               $(hmm_path)
               $(msa_path)`

        result = run_subprocess(cmd)
        isfile(hmm_path) || error("hmmbuild did not produce output HMM at $hmm_path")
        return read(hmm_path, String)
    end
end
