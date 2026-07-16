"""
chemical_components.jl — CCD (Chemical Component Dictionary) data structures and loader.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

"""
CcdAtom: one atom record from a CCD component.
"""
struct CcdAtom
    atom_id::String
    type_symbol::String
    charge::Float32
    x_ideal::Float32
    y_ideal::Float32
    z_ideal::Float32
    leaving_atom_flag::Bool
    aromatic_flag::Bool
end

"""
CcdBond: one bond record from a CCD component.
"""
struct CcdBond
    atom_id_1::String
    atom_id_2::String
    value_order::String   # "SING","DOUB","TRIP","AROM"
    aromatic_flag::Bool
end

"""
CcdComponent: all data for one CCD component (residue/ligand).
"""
struct CcdComponent
    comp_id::String
    name::String
    comp_type::String
    formula::String
    formula_weight::Float32
    atoms::Vector{CcdAtom}
    bonds::Vector{CcdBond}
end

"""
Ccd: the full Chemical Component Dictionary, optionally merged with user CCD entries.
Wraps a Dict{String, CcdComponent}.
"""
struct Ccd
    components::Dict{String, CcdComponent}
end

function Ccd(; user_ccd::Union{String,Nothing}=nothing)
    components = Dict{String, CcdComponent}()
    # Try to load the default CCD from a serialized binary file
    default_path = _get_default_ccd_path()
    if default_path !== nothing && isfile(default_path)
        _load_ccd_binary!(components, default_path)
    end
    # Merge user CCD entries parsed from CIF
    if user_ccd !== nothing
        _parse_user_ccd!(components, user_ccd)
    end
    return Ccd(components)
end

function Base.get(ccd::Ccd, key::String, default=nothing)
    return get(ccd.components, key, default)
end

function Base.haskey(ccd::Ccd, key::String)
    return haskey(ccd.components, key)
end

function Base.getindex(ccd::Ccd, key::String)
    return ccd.components[key]
end

function Base.keys(ccd::Ccd)
    return keys(ccd.components)
end

function Base.show(io::IO, ccd::Ccd)
    print(io, "Ccd($(length(ccd.components)) components)")
end

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

function _get_default_ccd_path()::Union{String,Nothing}
    # Check common locations for the CCD binary
    candidates = [
        joinpath(@__DIR__, "..", "..", "data", "ccd.bin"),
        joinpath(@__DIR__, "..", "..", "data", "ccd", "ccd.bin"),
        get(ENV, "CCD_PATH", ""),
    ]
    for p in candidates
        isfile(p) && return p
    end
    return nothing
end

function _load_ccd_binary!(components::Dict{String,CcdComponent}, path::String)
    @info "Loading CCD from $path"
    try
        open(path, "r") do io
            _deserialize_ccd!(components, io)
        end
    catch e
        @warn "Failed to load CCD from $path: $e"
    end
end

function _deserialize_ccd!(components::Dict{String,CcdComponent}, io::IO)
    # Simple binary format: count then serialized Julia objects
    # The format written by ccd_pickle_gen.jl
    try
        data = deserialize(io)
        if data isa Dict
            for (k, v) in data
                if v isa CcdComponent
                    components[k] = v
                end
            end
        end
    catch e
        @warn "CCD deserialization error: $e"
    end
end

function _parse_user_ccd!(components::Dict{String,CcdComponent}, cif_text::String)
    # Parse user-provided CIF text and merge components
    blocks = _split_ccd_cif_blocks(cif_text)
    for block in blocks
        comp = _parse_ccd_block(block)
        comp !== nothing && (components[comp.comp_id] = comp)
    end
end

function _split_ccd_cif_blocks(cif_text::String)::Vector{String}
    # Split on "data_XXXXX" lines
    lines = split(cif_text, '\n')
    blocks = String[]
    current = IOBuffer()
    in_block = false
    for line in lines
        if startswith(strip(line), "data_")
            if in_block
                push!(blocks, String(take!(current)))
                current = IOBuffer()
            end
            in_block = true
        end
        in_block && println(current, line)
    end
    in_block && push!(blocks, String(take!(current)))
    return blocks
end

function _parse_ccd_block(block_text::String)::Union{CcdComponent,Nothing}
    lines = split(block_text, '\n')
    isempty(lines) && return nothing

    # Extract comp_id from data_ line
    header = strip(lines[1])
    startswith(header, "data_") || return nothing
    comp_id = header[6:end]

    kv = Dict{String,String}()
    loops = Dict{String,Vector{Vector{String}}}()
    current_loop_keys = String[]
    current_loop_rows = Vector{Vector{String}}()
    in_loop = false
    in_value = false

    i = 2
    while i <= length(lines)
        line = strip(lines[i])
        if isempty(line) || startswith(line, '#')
            in_loop = false
            i += 1
            continue
        end
        if line == "loop_"
            if in_loop && !isempty(current_loop_keys)
                for key in current_loop_keys
                    loops[key] = [[r[j] for r in current_loop_rows]
                                  for j in 1:length(current_loop_keys)]
                end
            end
            current_loop_keys = String[]
            current_loop_rows = Vector{Vector{String}}()
            in_loop = true
            i += 1
            continue
        end
        if in_loop
            if startswith(line, '_')
                push!(current_loop_keys, split(line)[1])
                i += 1
            else
                # Data rows
                tokens = _tokenize_cif_line(line)
                if !isempty(tokens)
                    push!(current_loop_rows, tokens)
                end
                i += 1
            end
        else
            if startswith(line, '_')
                parts = split(line, limit=2)
                key = parts[1]
                if length(parts) == 2
                    kv[key] = strip(parts[2])
                else
                    # Multi-line value
                    i += 1
                    if i <= length(lines)
                        kv[key] = strip(lines[i])
                    end
                end
            end
            i += 1
        end
    end
    # Finalize last loop
    if in_loop && !isempty(current_loop_keys) && !isempty(current_loop_rows)
        ncols = length(current_loop_keys)
        for (j, key) in enumerate(current_loop_keys)
            loops[key] = [r[min(j,end)] for r in current_loop_rows]
        end
    end

    # Extract atom data
    atoms = CcdAtom[]
    atom_ids = get(loops, "_chem_comp_atom.atom_id", String[])
    type_syms = get(loops, "_chem_comp_atom.type_symbol", String[])
    charges  = get(loops, "_chem_comp_atom.charge", String[])
    x_ideals = get(loops, "_chem_comp_atom.pdbx_model_Cartn_x_ideal", String[])
    y_ideals = get(loops, "_chem_comp_atom.pdbx_model_Cartn_y_ideal", String[])
    z_ideals = get(loops, "_chem_comp_atom.pdbx_model_Cartn_z_ideal", String[])
    leaving  = get(loops, "_chem_comp_atom.pdbx_leaving_atom_flag", String[])
    aromatic = get(loops, "_chem_comp_atom.pdbx_aromatic_flag", String[])

    for k in 1:length(atom_ids)
        push!(atoms, CcdAtom(
            _stripquotes(get(atom_ids, k, "")),
            _stripquotes(get(type_syms, k, "C")),
            _parse_float32(get(charges, k, "0")),
            _parse_float32(get(x_ideals, k, "0")),
            _parse_float32(get(y_ideals, k, "0")),
            _parse_float32(get(z_ideals, k, "0")),
            get(leaving, k, "N") == "Y",
            get(aromatic, k, "N") == "Y",
        ))
    end

    # Extract bond data
    bonds = CcdBond[]
    b_atom1  = get(loops, "_chem_comp_bond.atom_id_1", String[])
    b_atom2  = get(loops, "_chem_comp_bond.atom_id_2", String[])
    b_order  = get(loops, "_chem_comp_bond.value_order", String[])
    b_arom   = get(loops, "_chem_comp_bond.pdbx_aromatic_flag", String[])
    for k in 1:length(b_atom1)
        push!(bonds, CcdBond(
            _stripquotes(get(b_atom1, k, "")),
            _stripquotes(get(b_atom2, k, "")),
            get(b_order, k, "SING"),
            get(b_arom, k, "N") == "Y",
        ))
    end

    name   = get(kv, "_chem_comp.name", comp_id)
    ctype  = get(kv, "_chem_comp.type", "NON-POLYMER")
    form   = get(kv, "_chem_comp.formula", "")
    fw_str = get(kv, "_chem_comp.formula_weight", "0")
    fw     = _parse_float32(fw_str)

    return CcdComponent(comp_id, _stripquotes(name), _stripquotes(ctype),
                        _stripquotes(form), fw, atoms, bonds)
end

function _tokenize_cif_line(line::String)::Vector{String}
    tokens = String[]
    i = 1
    while i <= length(line)
        c = line[i]
        if isspace(c)
            i += 1
        elseif c == '\'' || c == '"'
            delim = c
            j = i + 1
            while j <= length(line) && line[j] != delim
                j += 1
            end
            push!(tokens, line[i+1:j-1])
            i = j + 1
        else
            j = i
            while j <= length(line) && !isspace(line[j])
                j += 1
            end
            push!(tokens, line[i:j-1])
            i = j
        end
    end
    return tokens
end

function _stripquotes(s::String)::String
    s = strip(s)
    (startswith(s, "'") && endswith(s, "'")) && return s[2:end-1]
    (startswith(s, "\"") && endswith(s, "\"")) && return s[2:end-1]
    return s
end

function _parse_float32(s::String)::Float32
    try
        return parse(Float32, s)
    catch
        return 0f0
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Public query helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    get_component_atoms(ccd::Ccd, comp_id::String; include_hydrogens=false) -> Vector{CcdAtom}

Return atoms for a CCD component, optionally excluding hydrogens.
"""
function get_component_atoms(ccd::Ccd, comp_id::String;
                             include_hydrogens::Bool=false)::Vector{CcdAtom}
    comp = get(ccd, comp_id, nothing)
    comp === nothing && return CcdAtom[]
    if include_hydrogens
        return comp.atoms
    else
        return filter(a -> a.type_symbol != "H" && a.type_symbol != "D",
                      comp.atoms)
    end
end

"""
    get_component_bonds(ccd::Ccd, comp_id::String) -> Vector{CcdBond}
"""
function get_component_bonds(ccd::Ccd, comp_id::String)::Vector{CcdBond}
    comp = get(ccd, comp_id, nothing)
    return comp === nothing ? CcdBond[] : comp.bonds
end

"""
    get_ideal_positions(ccd::Ccd, comp_id::String; include_hydrogens=false)
        -> Dict{String, NTuple{3,Float32}}

Return atom_name => (x,y,z) ideal coordinates for a component.
"""
function get_ideal_positions(ccd::Ccd, comp_id::String;
                             include_hydrogens::Bool=false)::Dict{String,NTuple{3,Float32}}
    atoms = get_component_atoms(ccd, comp_id; include_hydrogens)
    return Dict(a.atom_id => (a.x_ideal, a.y_ideal, a.z_ideal) for a in atoms)
end
