"""
model_config.jl — Global model configuration structs.
"""

# ──────────────────────────────────────────────────────────────────────────────
# GlobalConfig
# ──────────────────────────────────────────────────────────────────────────────

"""
    GlobalConfig

Global architectural configuration for all model components.
"""
struct GlobalConfig
    bfloat16::Symbol   # :all, :none, :intermediate
    final_init::Symbol  # :zeros, :linear
    pair_attention_chunk_size::Vector{Tuple{Union{Int,Nothing},Union{Int,Nothing}}}
    pair_transition_shard_spec::Vector{Tuple{Union{Int,Nothing},Union{Int,Nothing}}}
    flash_attention_implementation::Symbol  # :triton, :cudnn, :xla
end

function GlobalConfig(;
    bfloat16::Symbol = :all,
    final_init::Symbol = :zeros,
    pair_attention_chunk_size = [(1536, 128), (nothing, 32)],
    pair_transition_shard_spec = [(2048, nothing), (nothing, 1024)],
    flash_attention_implementation::Symbol = :xla,
)
    return GlobalConfig(bfloat16, final_init,
                        Tuple{Union{Int,Nothing},Union{Int,Nothing}}[pair_attention_chunk_size...],
                        Tuple{Union{Int,Nothing},Union{Int,Nothing}}[pair_transition_shard_spec...],
                        flash_attention_implementation)
end

function Base.show(io::IO, c::GlobalConfig)
    print(io, "GlobalConfig(bfloat16=$(c.bfloat16), final_init=$(c.final_init))")
end

# Default global config
const DEFAULT_GLOBAL_CONFIG = GlobalConfig()

# ──────────────────────────────────────────────────────────────────────────────
# Model Config
# ──────────────────────────────────────────────────────────────────────────────

"""
    ModelConfig

Top-level model configuration.
"""
struct ModelConfig
    global_config::GlobalConfig
    num_recycles::Int
    return_embeddings::Bool
    return_distogram::Bool
    seq_channel::Int
    pair_channel::Int
    msa_channel::Int
    num_trunk_layers::Int
    num_confidence_layers::Int
end

function ModelConfig(;
    global_config::GlobalConfig = GlobalConfig(),
    num_recycles::Int = 10,
    return_embeddings::Bool = false,
    return_distogram::Bool = false,
    seq_channel::Int  = C_S,
    pair_channel::Int = C_Z,
    msa_channel::Int  = C_MSA,
    num_trunk_layers::Int = 48,
    num_confidence_layers::Int = 4,
)
    return ModelConfig(global_config, num_recycles, return_embeddings, return_distogram,
                       seq_channel, pair_channel, msa_channel,
                       num_trunk_layers, num_confidence_layers)
end

function Base.show(io::IO, c::ModelConfig)
    print(io, "ModelConfig(recycles=$(c.num_recycles), trunk=$(c.num_trunk_layers) layers)")
end

"""
    make_model_config(; kwargs...) -> ModelConfig

Create a ModelConfig with default or overridden settings.
"""
function make_model_config(; kwargs...)::ModelConfig
    return ModelConfig(; kwargs...)
end
