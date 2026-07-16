"""
msa_config.jl — Configuration structs for MSA search tools.
"""

using Dates

# ──────────────────────────────────────────────────────────────────────────────
# Database configuration
# ──────────────────────────────────────────────────────────────────────────────

struct DatabaseConfig
    name::String
    path::String
end

function Base.show(io::IO, c::DatabaseConfig)
    print(io, "DatabaseConfig($(c.name), $(c.path))")
end

# ──────────────────────────────────────────────────────────────────────────────
# Jackhmmer configuration
# ──────────────────────────────────────────────────────────────────────────────

struct JackhmmerConfig
    binary_path::String
    database_config::DatabaseConfig
    n_cpu::Int
    n_iter::Int
    e_value::Float64
    z_value::Union{Int,Nothing}
    dom_z_value::Union{Int,Nothing}
    max_sequences::Int
    max_parallel_shards::Union{Int,Nothing}
end

function JackhmmerConfig(;
    binary_path::String    = "jackhmmer",
    database_config::DatabaseConfig,
    n_cpu::Int             = 8,
    n_iter::Int            = 1,
    e_value::Float64       = 1e-4,
    z_value::Union{Int,Nothing}     = nothing,
    dom_z_value::Union{Int,Nothing} = nothing,
    max_sequences::Int     = 10000,
    max_parallel_shards::Union{Int,Nothing} = nothing,
)
    return JackhmmerConfig(binary_path, database_config, n_cpu, n_iter,
                           e_value, z_value, dom_z_value, max_sequences,
                           max_parallel_shards)
end

# ──────────────────────────────────────────────────────────────────────────────
# Nhmmer configuration
# ──────────────────────────────────────────────────────────────────────────────

struct NhmmerConfig
    binary_path::String
    hmmalign_binary_path::String
    hmmbuild_binary_path::String
    database_config::DatabaseConfig
    n_cpu::Int
    e_value::Float64
    alphabet::String
    z_value::Union{Float64,Nothing}
    max_sequences::Int
    max_parallel_shards::Union{Int,Nothing}
end

function NhmmerConfig(;
    binary_path::String            = "nhmmer",
    hmmalign_binary_path::String   = "hmmalign",
    hmmbuild_binary_path::String   = "hmmbuild",
    database_config::DatabaseConfig,
    n_cpu::Int                     = 8,
    e_value::Float64               = 1e-3,
    alphabet::String               = "rna",
    z_value::Union{Float64,Nothing} = nothing,
    max_sequences::Int             = 10000,
    max_parallel_shards::Union{Int,Nothing} = nothing,
)
    return NhmmerConfig(binary_path, hmmalign_binary_path, hmmbuild_binary_path,
                        database_config, n_cpu, e_value, alphabet, z_value,
                        max_sequences, max_parallel_shards)
end

# ──────────────────────────────────────────────────────────────────────────────
# Template search configuration
# ──────────────────────────────────────────────────────────────────────────────

struct TemplateToolConfig
    database_path::String
    chain_poly_type::String
    hmmsearch_config::HmmsearchConfig
end

struct TemplateFilterConfig
    max_subsequence_ratio::Float64
    min_align_ratio::Float64
    min_hit_length::Int
    deduplicate_sequences::Bool
    max_hits::Int
    max_template_date::Date
end

function TemplateFilterConfig(;
    max_subsequence_ratio::Float64 = 0.95,
    min_align_ratio::Float64       = 0.1,
    min_hit_length::Int            = 10,
    deduplicate_sequences::Bool    = true,
    max_hits::Int                  = 20,
    max_template_date::Date        = Date(2021, 9, 30),
)
    return TemplateFilterConfig(max_subsequence_ratio, min_align_ratio,
                                min_hit_length, deduplicate_sequences,
                                max_hits, max_template_date)
end

struct TemplatesConfig
    template_tool_config::TemplateToolConfig
    filter_config::TemplateFilterConfig
end

# ──────────────────────────────────────────────────────────────────────────────
# Run configuration (combines a tool config with chain type and crop size)
# ──────────────────────────────────────────────────────────────────────────────

struct RunConfig
    config::Union{JackhmmerConfig,NhmmerConfig}
    chain_poly_type::String
    crop_size::Union{Int,Nothing}
end

function RunConfig(; config, chain_poly_type::String, crop_size=nothing)
    return RunConfig(config, chain_poly_type, crop_size)
end
