"""
Base configuration types and utilities shared across the pipeline.
"""

"""
    BaseConfig

Abstract supertype for all configuration structs. Provides common
serialisation/deserialisation helpers.
"""
abstract type BaseConfig end

"""
    config_to_dict(cfg::BaseConfig) -> Dict{String,Any}

Convert a configuration struct to a plain dictionary for JSON serialisation.
"""
function config_to_dict(cfg::T)::Dict{String,Any} where {T<:BaseConfig}
    d = Dict{String,Any}()
    for field in fieldnames(T)
        val = getfield(cfg, field)
        if val isa BaseConfig
            d[String(field)] = config_to_dict(val)
        elseif val isa Vector && !isempty(val) && first(val) isa BaseConfig
            d[String(field)] = [config_to_dict(v) for v in val]
        else
            d[String(field)] = val
        end
    end
    return d
end

"""
    validate_positive_int(val::Int, name::String)

Raise ArgumentError if val < 1.
"""
function validate_positive_int(val::Int, name::String)
    if val < 1
        throw(ArgumentError("$name must be >= 1, got $val"))
    end
end

"""
    validate_non_negative_int(val::Int, name::String)

Raise ArgumentError if val < 0.
"""
function validate_non_negative_int(val::Int, name::String)
    if val < 0
        throw(ArgumentError("$name must be >= 0, got $val"))
    end
end

"""
    validate_strictly_increasing(v::Vector{Int}, name::String)

Raise ArgumentError unless v is strictly increasing.
"""
function validate_strictly_increasing(v::Vector{Int}, name::String)
    for i in 2:length(v)
        if v[i] <= v[i-1]
            throw(ArgumentError("$name must be strictly increasing, got $v"))
        end
    end
end

"""
    expand_env_vars(s::String) -> String

Expand environment variable references of the form \$VARNAME in s.
"""
function expand_env_vars(s::String)::String
    return replace(s, r"\$([A-Z_][A-Z0-9_]*)" => (m) -> get(ENV, m[2:end], m))
end
