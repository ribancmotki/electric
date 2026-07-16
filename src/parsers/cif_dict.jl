"""
cif_dict.jl — Robust mmCIF / PDBx CIF parser producing a nested Dict.

The parser handles:
  - Scalar key-value pairs: _key value
  - Multi-line string values (semicolon-delimited)
  - loop_ blocks with multiple columns
  - Quoted strings (single and double quotes)
  - Comments (#)
"""

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    CifDict

Represents the parsed content of a single CIF data block.
Contains key-value pairs and loop tables.

Fields:
- `name::String`: the data block name (after `data_`).
- `scalars::Dict{String,String}`: non-loop key-value pairs.
- `loops::Dict{String,Vector{String}}`: loop columns; key is the column tag,
  value is the flat column vector (row-major order, all columns same length).
"""
struct CifDict
    name::String
    scalars::Dict{String,String}
    loops::Dict{String,Vector{String}}
end

function Base.show(io::IO, d::CifDict)
    print(io, "CifDict(\"$(d.name)\", scalars=$(length(d.scalars)), loop_keys=$(length(d.loops)))")
end

function Base.get(d::CifDict, key::String, default=nothing)
    k = lowercase(key)
    haskey(d.scalars, k) && return d.scalars[k]
    haskey(d.loops, k)   && return d.loops[k]
    return default
end

function Base.haskey(d::CifDict, key::String)
    k = lowercase(key)
    return haskey(d.scalars, k) || haskey(d.loops, k)
end

"""
    parse_cif(text::String) -> Vector{CifDict}

Parse a CIF string and return one CifDict per data block.
"""
function parse_cif(text::String)::Vector{CifDict}
    tokens = _tokenize_cif(text)
    return _parse_tokens(tokens)
end

"""
    parse_cif_first_block(text::String) -> CifDict

Parse a CIF string and return only the first data block.
Throws an error if the string is empty.
"""
function parse_cif_first_block(text::String)::CifDict
    blocks = parse_cif(text)
    isempty(blocks) && error("CIF string contains no data blocks")
    return blocks[1]
end

# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────

struct _Token
    type::Symbol   # :data, :loop, :key, :value, :semicolon_str
    value::String
end

function _tokenize_cif(text::String)::Vector{_Token}
    tokens = _Token[]
    lines = split(text, '\n')
    i = 1
    while i <= length(lines)
        line = lines[i]
        # Strip inline comments but be careful about quoted strings
        stripped = _strip_comment(line)
        s = strip(stripped)
        if isempty(s)
            i += 1
            continue
        end
        if startswith(s, "data_")
            push!(tokens, _Token(:data, s[6:end]))
            i += 1
        elseif s == "loop_"
            push!(tokens, _Token(:loop, ""))
            i += 1
        elseif startswith(s, "_")
            # Could be "key value" on one line or just "key"
            space_idx = findfirst(isspace, s)
            if space_idx !== nothing
                key = s[1:space_idx-1]
                rest = strip(s[space_idx+1:end])
                push!(tokens, _Token(:key, lowercase(key)))
                if !isempty(rest) && !startswith(rest, "#")
                    push!(tokens, _Token(:value, _unquote(rest)))
                end
            else
                push!(tokens, _Token(:key, lowercase(s)))
            end
            i += 1
        elseif startswith(s, ";")
            # Multi-line semicolon-delimited string
            parts = String[]
            i += 1
            while i <= length(lines)
                l = lines[i]
                stripped_l = strip(l)
                if stripped_l == ";"
                    i += 1
                    break
                end
                push!(parts, l)
                i += 1
            end
            push!(tokens, _Token(:semicolon_str, join(parts, "\n")))
        elseif startswith(s, "#")
            i += 1
            continue
        else
            # Bare value or quoted string
            inline_vals = _extract_values(s)
            for v in inline_vals
                push!(tokens, _Token(:value, v))
            end
            i += 1
        end
    end
    return tokens
end

function _strip_comment(line::String)::String
    # Remove trailing comments, but not inside quoted strings
    result = IOBuffer()
    in_sq = false
    in_dq = false
    for ch in line
        if ch == '\'' && !in_dq; in_sq = !in_sq
        elseif ch == '"' && !in_sq; in_dq = !in_dq
        elseif ch == '#' && !in_sq && !in_dq; break
        end
        print(result, ch)
    end
    return String(take!(result))
end

function _unquote(s::String)::String
    s = strip(s)
    if (startswith(s, "'") && endswith(s, "'") && length(s) >= 2) ||
       (startswith(s, "\"") && endswith(s, "\"") && length(s) >= 2)
        return s[2:end-1]
    end
    return s
end

function _extract_values(s::String)::Vector{String}
    values = String[]
    i = 1
    while i <= length(s)
        c = s[i]
        if isspace(c)
            i += 1
            continue
        end
        if c == '\'' || c == '"'
            delim = c
            j = i + 1
            buf = IOBuffer()
            while j <= length(s)
                if s[j] == delim && (j == length(s) || isspace(s[j+1]))
                    break
                end
                print(buf, s[j])
                j += 1
            end
            push!(values, String(take!(buf)))
            i = j + 1
        else
            j = i
            while j <= length(s) && !isspace(s[j])
                j += 1
            end
            push!(values, s[i:j-1])
            i = j
        end
    end
    return values
end

# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

function _parse_tokens(tokens::Vector{_Token})::Vector{CifDict}
    blocks = CifDict[]
    i = 1
    while i <= length(tokens)
        t = tokens[i]
        if t.type == :data
            i, block = _parse_block(tokens, i + 1, t.value)
            push!(blocks, block)
        else
            i += 1
        end
    end
    return blocks
end

function _parse_block(tokens::Vector{_Token}, start::Int, name::String)
    scalars = Dict{String,String}()
    loops   = Dict{String,Vector{String}}()
    i = start
    while i <= length(tokens)
        t = tokens[i]
        if t.type == :data
            # New block starts
            return i, CifDict(name, scalars, loops)
        elseif t.type == :loop
            i, cols, data = _parse_loop(tokens, i + 1)
            n = length(data)
            ncols = length(cols)
            if ncols > 0 && n > 0
                for (j, col) in enumerate(cols)
                    loops[col] = [data[k] for k in j:ncols:n]
                end
            end
        elseif t.type == :key
            key = t.value
            i += 1
            if i <= length(tokens) && tokens[i].type in (:value, :semicolon_str)
                scalars[key] = tokens[i].value
                i += 1
            else
                scalars[key] = "."
            end
        else
            i += 1
        end
    end
    return i, CifDict(name, scalars, loops)
end

function _parse_loop(tokens::Vector{_Token}, start::Int)
    cols = String[]
    data = String[]
    i = start
    # Collect column keys
    while i <= length(tokens) && tokens[i].type == :key
        push!(cols, tokens[i].value)
        i += 1
    end
    # Collect data values
    while i <= length(tokens) && tokens[i].type in (:value, :semicolon_str)
        push!(data, tokens[i].value)
        i += 1
    end
    return i, cols, data
end

# ──────────────────────────────────────────────────────────────────────────────
# Convenience getters
# ──────────────────────────────────────────────────────────────────────────────

"""
    get_scalar(d::CifDict, key::String, default::String="") -> String

Get a scalar value from a CifDict, with fallback to default.
"""
function get_scalar(d::CifDict, key::String, default::String="")::String
    k = lowercase(key)
    return get(d.scalars, k, default)
end

"""
    get_loop_col(d::CifDict, key::String) -> Vector{String}

Get a loop column from a CifDict. Returns empty vector if not found.
"""
function get_loop_col(d::CifDict, key::String)::Vector{String}
    k = lowercase(key)
    return get(d.loops, k, String[])
end

"""
    _get_loop_col(block_dict::Dict{String,Any}, key::String, default) -> Any

Get a loop column from a raw Dict representation (legacy API).
Returns default if not found.
"""
function _get_loop_col(block_dict::Dict{String,Any}, key::String, default)
    k = lowercase(key)
    return get(block_dict, k, default)
end
