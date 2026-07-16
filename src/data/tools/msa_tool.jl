"""
msa_tool.jl — Unified interface for MSA search tools.
"""

"""
    MsaTool

Abstract type for MSA search tools.
"""
abstract type MsaTool end

"""
    JackhmmerTool <: MsaTool

Wrapper for jackhmmer-based protein MSA search.
"""
struct JackhmmerTool <: MsaTool
    config::JackhmmerConfig
end

"""
    NhmmerTool <: MsaTool

Wrapper for nhmmer-based RNA/DNA MSA search.
"""
struct NhmmerTool <: MsaTool
    config::NhmmerConfig
end

"""
    run_msa_search(tool::MsaTool, sequence::String, chain_poly_type::String) -> String

Run MSA search for a given sequence, returning an A3M string.
"""
function run_msa_search(tool::JackhmmerTool, sequence::String, chain_poly_type::String)::String
    query_fasta = make_single_record_fasta(sequence)
    return run_jackhmmer(tool.config, query_fasta)
end

function run_msa_search(tool::NhmmerTool, sequence::String, chain_poly_type::String)::String
    query_fasta = make_single_record_fasta(sequence)
    return run_nhmmer(tool.config, query_fasta)
end

"""
    get_msa_for_sequence(sequence::String, tools::Vector{<:MsaTool},
                         chain_poly_type::String; deduplicate=true) -> String

Run MSA search with all tools and merge results.
"""
function get_msa_for_sequence(sequence::String, tools::Vector{<:MsaTool},
                               chain_poly_type::String;
                               deduplicate::Bool=true)::String
    results = String[]
    for tool in tools
        a3m = run_msa_search(tool, sequence, chain_poly_type)
        !isempty(a3m) && push!(results, a3m)
    end
    isempty(results) && return ">query\n$sequence\n"
    return merge_jackhmmer_results(results)
end
