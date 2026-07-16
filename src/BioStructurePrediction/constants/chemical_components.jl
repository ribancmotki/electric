"""
Chemical Components Dictionary (CCD) database interface.
The database is compiled by build_data.jl and loaded at runtime.
"""

using Dates
using Serialization

"""
    CcdAtom

Per-atom data from the CCD for a single component.
"""
struct CcdAtom
    atom_id::String
    element::String
    charge::Float32
    ideal_x::Float32
    ideal_y::Float32
    ideal_z::Float32
    model_x::Float32
    model_y::Float32
    model_z::Float32
    leaving_atom::Bool
end

"""
    CcdBond

Bond data from the CCD for a single component.
"""
struct CcdBond
    atom_id_1::String
    atom_id_2::String
    bond_order::Int
    is_aromatic::Bool
end

"""
    CcdComponent

All data for a single chemical component from the CCD.
"""
struct CcdComponent
    comp_id::String
    formula::String
    name::String
    type::String
    mon_nstd_flag::Bool
    release_date::Union{Date,Nothing}
    atoms::Vector{CcdAtom}
    bonds::Vector{CcdBond}
end

"""
    Ccd

The full Chemical Components Dictionary, optionally extended with user-defined components.
"""
struct Ccd
    components::Dict{String,CcdComponent}
    ref_max_modified_date::Union{Date,Nothing}

    function Ccd(
        components::Dict{String,CcdComponent};
        user_ccd::Union{String,Nothing} = nothing,
        ref_max_modified_date::Union{Date,Nothing} = nothing
    )
        all_components = copy(components)
        if user_ccd !== nothing
            user_comps = parse_ccd_mmcif(user_ccd)
            merge!(all_components, user_comps)
        end
        new(all_components, ref_max_modified_date)
    end
end

"""
    load_ccd(; user_ccd=nothing, ref_max_modified_date=nothing) -> Ccd

Load the compiled CCD database from the package data directory.
"""
function load_ccd(;
    user_ccd::Union{String,Nothing} = nothing,
    ref_max_modified_date::Union{Date,Nothing} = nothing
)::Ccd
    db_path = get_ccd_database_path()
    if !isfile(db_path)
        @warn "CCD database not found at $db_path. Run build_data.jl first. Using empty CCD."
        return Ccd(Dict{String,CcdComponent}(); user_ccd=user_ccd, ref_max_modified_date=ref_max_modified_date)
    end
    components = open(deserialize, db_path)::Dict{String,CcdComponent}
    return Ccd(components; user_ccd=user_ccd, ref_max_modified_date=ref_max_modified_date)
end

"""
    get_component(ccd::Ccd, comp_id::String) -> Union{CcdComponent,Nothing}

Look up a component by its CCD code. Returns nothing if not found.
"""
function get_component(ccd::Ccd, comp_id::String)::Union{CcdComponent,Nothing}
    return get(ccd.components, comp_id, nothing)
end

"""
    get_component!(ccd::Ccd, comp_id::String) -> CcdComponent

Look up a component by CCD code, raising an error if not found.
"""
function get_component!(ccd::Ccd, comp_id::String)::CcdComponent
    comp = get_component(ccd, comp_id)
    if comp === nothing
        error("CCD component '$comp_id' not found. Provide it via --userCCD or userCCDPath.")
    end
    return comp
end

"""
    get_component_atoms(ccd::Ccd, comp_id::String) -> NamedTuple

Return atom names, elements, charges, ideal coordinates, model coordinates, and bond connectivity
for the given CCD component.
"""
function get_component_atoms(ccd::Ccd, comp_id::String)::NamedTuple
    comp = get_component!(ccd, comp_id)
    non_leaving = filter(a -> !a.leaving_atom, comp.atoms)
    atom_names     = [a.atom_id  for a in non_leaving]
    elements       = [a.element  for a in non_leaving]
    charges        = Float32[a.charge   for a in non_leaving]
    ideal_coords   = Float32[a.ideal_x  a.ideal_y  a.ideal_z; [row for row in [[a.ideal_x, a.ideal_y, a.ideal_z] for a in non_leaving]]...]
    # Build properly shaped arrays
    n = length(non_leaving)
    ideal_pos  = zeros(Float32, n, 3)
    model_pos  = zeros(Float32, n, 3)
    for (i, a) in enumerate(non_leaving)
        ideal_pos[i, 1] = a.ideal_x
        ideal_pos[i, 2] = a.ideal_y
        ideal_pos[i, 3] = a.ideal_z
        model_pos[i, 1] = a.model_x
        model_pos[i, 2] = a.model_y
        model_pos[i, 3] = a.model_z
    end
    # Filter bonds to only non-leaving atoms
    non_leaving_set = Set(atom_names)
    bonds = filter(b -> b.atom_id_1 in non_leaving_set && b.atom_id_2 in non_leaving_set, comp.bonds)
    return (
        atom_names = atom_names,
        elements   = elements,
        charges    = charges,
        ideal_pos  = ideal_pos,
        model_pos  = model_pos,
        bonds      = bonds,
        release_date = comp.release_date,
    )
end

"""
    has_ideal_coordinates(ccd::Ccd, comp_id::String) -> Bool

Return true if the component has non-zero ideal coordinates.
"""
function has_ideal_coordinates(ccd::Ccd, comp_id::String)::Bool
    comp = get_component(ccd, comp_id)
    comp === nothing && return false
    for atom in comp.atoms
        if !atom.leaving_atom && (atom.ideal_x != 0 || atom.ideal_y != 0 || atom.ideal_z != 0)
            return true
        end
    end
    return false
end

"""
    has_model_coordinates(ccd::Ccd, comp_id::String) -> Bool

Return true if the component has non-zero model coordinates.
"""
function has_model_coordinates(ccd::Ccd, comp_id::String)::Bool
    comp = get_component(ccd, comp_id)
    comp === nothing && return false
    for atom in comp.atoms
        if !atom.leaving_atom && (atom.model_x != 0 || atom.model_y != 0 || atom.model_z != 0)
            return true
        end
    end
    return false
end

"""
    component_before_date(ccd::Ccd, comp_id::String, date::Date) -> Bool

Return true if the component's release date is before or on the given date,
or if it has no release date.
"""
function component_before_date(ccd::Ccd, comp_id::String, cutoff_date::Date)::Bool
    comp = get_component(ccd, comp_id)
    comp === nothing && return false
    comp.release_date === nothing && return true
    return comp.release_date <= cutoff_date
end

"""
    parse_ccd_mmcif(mmcif_text::String) -> Dict{String,CcdComponent}

Parse CCD-format mmCIF text into a dict of CcdComponent objects.
Handles multi-block mmCIF files (one block per component).
"""
function parse_ccd_mmcif(mmcif_text::String)::Dict{String,CcdComponent}
    components = Dict{String,CcdComponent}()
    # Split into data blocks
    blocks = split(mmcif_text, r"(?=^data_)"m)
    for block in blocks
        block = strip(block)
        isempty(block) && continue
        comp = parse_ccd_block(block)
        comp !== nothing && (components[comp.comp_id] = comp)
    end
    return components
end

"""
    parse_ccd_block(block::String) -> Union{CcdComponent,Nothing}

Parse a single mmCIF data block for one CCD component.
"""
function parse_ccd_block(block::String)::Union{CcdComponent,Nothing}
    lines = split(block, '\n')
    isempty(lines) && return nothing

    # Extract comp_id from data_ line
    header_match = match(r"^data_(\S+)", lines[1])
    header_match === nothing && return nothing
    comp_id = String(header_match.captures[1])

    # Parse simple key-value pairs and loop_ tables
    kv = Dict{String,String}()
    atoms = CcdAtom[]
    bonds = CcdBond[]

    i = 1
    while i <= length(lines)
        line = strip(lines[i])
        if startswith(line, "_chem_comp.") && !startswith(line, "loop_")
            parts = split(line)
            if length(parts) >= 2
                kv[parts[1]] = parts[2]
            end
            i += 1
        elseif line == "loop_"
            # Read loop headers
            i += 1
            headers = String[]
            while i <= length(lines) && startswith(strip(lines[i]), "_")
                push!(headers, strip(strip(lines[i])))
                i += 1
            end
            # Read loop data
            if !isempty(headers) && startswith(first(headers), "_chem_comp_atom.")
                # Parse atom loop
                while i <= length(lines)
                    data_line = strip(lines[i])
                    isempty(data_line) && (i += 1; break)
                    startswith(data_line, "#") && (i += 1; break)
                    startswith(data_line, "_") && break
                    startswith(data_line, "loop_") && break
                    startswith(data_line, "data_") && break
                    row = parse_mmcif_row(data_line)
                    if length(row) >= length(headers)
                        atom = parse_ccd_atom_row(headers, row)
                        atom !== nothing && push!(atoms, atom)
                    end
                    i += 1
                end
            elseif !isempty(headers) && startswith(first(headers), "_chem_comp_bond.")
                # Parse bond loop
                while i <= length(lines)
                    data_line = strip(lines[i])
                    isempty(data_line) && (i += 1; break)
                    startswith(data_line, "#") && (i += 1; break)
                    startswith(data_line, "_") && break
                    startswith(data_line, "loop_") && break
                    startswith(data_line, "data_") && break
                    row = parse_mmcif_row(data_line)
                    if length(row) >= length(headers)
                        bond = parse_ccd_bond_row(headers, row)
                        bond !== nothing && push!(bonds, bond)
                    end
                    i += 1
                end
            else
                # Skip other loops
                while i <= length(lines)
                    data_line = strip(lines[i])
                    if isempty(data_line) || startswith(data_line, "#") ||
                       startswith(data_line, "_") || startswith(data_line, "loop_") ||
                       startswith(data_line, "data_")
                        break
                    end
                    i += 1
                end
            end
        else
            i += 1
        end
    end

    formula    = get(kv, "_chem_comp.formula", "")
    name       = get(kv, "_chem_comp.name", "")
    comp_type  = get(kv, "_chem_comp.type", "")
    mon_flag   = get(kv, "_chem_comp.mon_nstd_flag", "n") == "y"
    date_str   = get(kv, "_chem_comp.pdbx_initial_date", "")
    release_date = if !isempty(date_str) && date_str != "?"
        try Date(date_str, dateformat"yyyy-mm-dd") catch; nothing end
    else
        nothing
    end

    return CcdComponent(comp_id, formula, name, comp_type, mon_flag, release_date, atoms, bonds)
end

function parse_mmcif_row(line::String)::Vector{String}
    tokens = String[]
    i = 1
    s = line
    n = length(s)
    while i <= n
        c = s[i]
        if c == '\'' || c == '"'
            # Quoted string
            quote_char = c
            i += 1
            start = i
            while i <= n && s[i] != quote_char
                i += 1
            end
            push!(tokens, s[start:i-1])
            i += 1
        elseif c == ' ' || c == '\t'
            i += 1
        else
            start = i
            while i <= n && s[i] != ' ' && s[i] != '\t'
                i += 1
            end
            push!(tokens, s[start:i-1])
        end
    end
    return tokens
end

function parse_ccd_atom_row(headers::Vector{String}, row::Vector{String})::Union{CcdAtom,Nothing}
    hmap = Dict(h => i for (i,h) in enumerate(headers))
    get_field(name, default="") = haskey(hmap, name) ? get(row, hmap[name], default) : default
    parse_float(s) = try parse(Float32, s) catch; 0f0 end

    atom_id  = get_field("_chem_comp_atom.atom_id")
    element  = get_field("_chem_comp_atom.type_symbol")
    charge   = parse_float(get_field("_chem_comp_atom.charge", "0"))
    ideal_x  = parse_float(get_field("_chem_comp_atom.pdbx_model_Cartn_x_ideal", "0"))
    ideal_y  = parse_float(get_field("_chem_comp_atom.pdbx_model_Cartn_y_ideal", "0"))
    ideal_z  = parse_float(get_field("_chem_comp_atom.pdbx_model_Cartn_z_ideal", "0"))
    model_x  = parse_float(get_field("_chem_comp_atom.model_Cartn_x", "0"))
    model_y  = parse_float(get_field("_chem_comp_atom.model_Cartn_y", "0"))
    model_z  = parse_float(get_field("_chem_comp_atom.model_Cartn_z", "0"))
    leaving  = get_field("_chem_comp_atom.pdbx_leaving_atom_flag", "N") == "Y"

    isempty(atom_id) && return nothing
    return CcdAtom(atom_id, element, charge, ideal_x, ideal_y, ideal_z, model_x, model_y, model_z, leaving)
end

function parse_ccd_bond_row(headers::Vector{String}, row::Vector{String})::Union{CcdBond,Nothing}
    hmap = Dict(h => i for (i,h) in enumerate(headers))
    get_field(name, default="") = haskey(hmap, name) ? get(row, hmap[name], default) : default

    atom1    = get_field("_chem_comp_bond.atom_id_1")
    atom2    = get_field("_chem_comp_bond.atom_id_2")
    order_s  = get_field("_chem_comp_bond.value_order", "SING")
    aromatic = get_field("_chem_comp_bond.pdbx_aromatic_flag", "N") == "Y"

    isempty(atom1) || isempty(atom2) && return nothing
    bond_order = get(BOND_ORDER_MAP, order_s, 1)
    return CcdBond(atom1, atom2, bond_order, aromatic)
end
