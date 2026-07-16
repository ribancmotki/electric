"""
base_config.jl — Base configuration types and utilities.
"""

using JSON3

# ──────────────────────────────────────────────────────────────────────────────
# Abstract base type for all configuration structs
# ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractConfig

Abstract base for all configuration structs. Subtypes can be serialized to/from JSON.
"""
abstract type AbstractConfig end

# ──────────────────────────────────────────────────────────────────────────────
# JSON serialization support
# ──────────────────────────────────────────────────────────────────────────────

"""
    config_to_dict(cfg::AbstractConfig) -> Dict{Symbol,Any}

Recursively convert a config struct to a nested Dict for JSON serialization.
"""
function config_to_dict(cfg::AbstractConfig)::Dict{Symbol,Any}
    d = Dict{Symbol,Any}()
    for field in fieldnames(typeof(cfg))
        val = getfield(cfg, field)
        if val isa AbstractConfig
            d[field] = config_to_dict(val)
        elseif val isa Vector && !isempty(val) && first(val) isa AbstractConfig
            d[field] = map(config_to_dict, val)
        else
            d[field] = val
        end
    end
    return d
end

"""
    config_to_json(cfg::AbstractConfig) -> String

Serialize a config to a JSON string.
"""
function config_to_json(cfg::AbstractConfig)::String
    return JSON3.write(config_to_dict(cfg))
end

# ──────────────────────────────────────────────────────────────────────────────
# Environment variable expansion
# ──────────────────────────────────────────────────────────────────────────────

"""
    expand_env_vars(s::String) -> String

Expand environment variable references in a string.
Supports both `\$VAR` and `\${VAR}` syntax.
"""
function expand_env_vars(s::String)::String
    # Replace ${VAR} patterns
    result = replace(s, r"\$\{([^}]+)\}" => m -> get(ENV, m[3:end-1], m))
    # Replace $VAR patterns (word boundary)
    result = replace(result, r"\$([A-Za-z_][A-Za-z0-9_]*)" => m -> get(ENV, m[2:end], m))
    return result
end

# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    validate_positive(val, name::String)

Assert that a numeric value is positive.
"""
function validate_positive(val, name::String)
    @assert val > 0 "Config field '$name' must be positive, got $val"
end

"""
    validate_range(val, lo, hi, name::String)

Assert that a value is in the closed interval [lo, hi].
"""
function validate_range(val, lo, hi, name::String)
    @assert lo <= val <= hi "Config field '$name' must be in [$lo, $hi], got $val"
end

"""
    validate_path_exists(path::String, name::String)

Assert that a filesystem path exists.
"""
function validate_path_exists(path::String, name::String)
    @assert isfile(path) || isdir(path) "Config field '$name': path '$path' does not exist"
end

# ──────────────────────────────────────────────────────────────────────────────
# Paths helper
# ──────────────────────────────────────────────────────────────────────────────

"""
    maybe_expand_path(path::Union{String,Nothing}) -> Union{String,Nothing}

Expand environment variables in a path, or return nothing unchanged.
"""
function maybe_expand_path(path::Union{String,Nothing})::Union{String,Nothing}
    path === nothing && return nothing
    return expand_env_vars(path)
end
