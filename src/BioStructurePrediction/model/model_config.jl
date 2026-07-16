"""
Model configuration for inference.
"""

"""
    ModelConfig

Configuration struct for the biomolecular structure prediction model.
"""
struct ModelConfig
    flash_attention_implementation::String  # "triton", "cudnn", or "xla"
    num_diffusion_samples::Int
    num_recycles::Int
    return_embeddings::Bool
    return_distogram::Bool
end

"""
    make_model_config(; kwargs...) -> ModelConfig

Construct a ModelConfig with named keyword arguments.
"""
function make_model_config(;
    flash_attention_implementation::String = "triton",
    num_diffusion_samples::Int             = 5,
    num_recycles::Int                      = 10,
    return_embeddings::Bool                = false,
    return_distogram::Bool                 = false,
)::ModelConfig
    flash_attention_implementation ∈ ("triton", "cudnn", "xla") ||
        error("flash_attention_implementation must be one of triton/cudnn/xla, got: $flash_attention_implementation")
    num_diffusion_samples >= 1 ||
        error("num_diffusion_samples must be >= 1, got $num_diffusion_samples")
    num_recycles >= 1 ||
        error("num_recycles must be >= 1, got $num_recycles")
    return ModelConfig(
        flash_attention_implementation,
        num_diffusion_samples,
        num_recycles,
        return_embeddings,
        return_distogram,
    )
end
