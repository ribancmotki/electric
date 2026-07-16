"""
hmmsearch.jl — Julia wrapper for hmmsearch.
"""

"""
HmmsearchConfig: configuration for hmmsearch.
"""
struct HmmsearchConfig
    hmmsearch_binary_path::String
    hmmbuild_binary_path::String
    filter_f1::Float64
    filter_f2::Float64
    filter_f3::Float64
    e_value::Float64
    inc_e::Float64
    dom_e::Float64
    incdom_e::Float64
    alphabet::String
end

function HmmsearchConfig(;
    hmmsearch_binary_path::String = "hmmsearch",
    hmmbuild_binary_path::String  = "hmmbuild",
    filter_f1::Float64   = 0.0005,
    filter_f2::Float64   = 0.00005,
    filter_f3::Float64   = 0.000005,
    e_value::Float64     = 0.001,
    inc_e::Float64       = 1e-5,
    dom_e::Float64       = 0.001,
    incdom_e::Float64    = 1e-5,
    alphabet::String     = "amino",
)
    return HmmsearchConfig(
        hmmsearch_binary_path, hmmbuild_binary_path,
        filter_f1, filter_f2, filter_f3,
        e_value, inc_e, dom_e, incdom_e, alphabet,
    )
end

"""
    run_hmmsearch(config::HmmsearchConfig, hmm_profile::String,
                  database_path::String) -> String

Run hmmsearch against a database, return tab-separated hits.
"""
function run_hmmsearch(config::HmmsearchConfig, hmm_profile::String,
                       database_path::String)::String
    mktempdir_cleanup() do tmpdir
        hmm_path  = joinpath(tmpdir, "query.hmm")
        tblout    = joinpath(tmpdir, "hits.tblout")
        domtblout = joinpath(tmpdir, "hits.domtblout")
        write(hmm_path, hmm_profile)

        cmd = `$(config.hmmsearch_binary_path)
               --F1 $(config.filter_f1)
               --F2 $(config.filter_f2)
               --F3 $(config.filter_f3)
               -E  $(config.e_value)
               --incE $(config.inc_e)
               --domE $(config.dom_e)
               --incdomE $(config.incdom_e)
               --noali
               --tblout $(tblout)
               --domtblout $(domtblout)
               -o /dev/null
               $(hmm_path)
               $(database_path)`

        run_subprocess(cmd)
        isfile(tblout) || return ""
        return read(tblout, String)
    end
end

"""
    parse_hmmsearch_tblout(tblout::String) -> Vector{Dict{String,Any}}

Parse hmmsearch tblout format.
"""
function parse_hmmsearch_tblout(tblout::String)::Vector{Dict{String,Any}}
    hits = Dict{String,Any}[]
    for line in eachline(IOBuffer(tblout))
        startswith(line, '#') && continue
        isempty(strip(line)) && continue
        cols = split(line)
        length(cols) < 6 && continue
        push!(hits, Dict{String,Any}(
            "target_name"  => cols[1],
            "query_name"   => cols[3],
            "e_value"      => parse(Float64, cols[5]),
            "score"        => parse(Float64, cols[6]),
            "description"  => length(cols) >= 19 ? join(cols[19:end], " ") : "",
        ))
    end
    return hits
end
