"""
parsing.jl — High-level structure parsing from files and strings.
"""

"""
    parse_structure_from_mmcif_string(mmcif_str::String; kwargs...) -> Structure

Parse a Structure from an mmCIF string. Accepts the same keyword arguments as `from_mmcif`.
"""
function parse_structure_from_mmcif_string(mmcif_str::String; kwargs...)::Structure
    return from_mmcif(mmcif_str; kwargs...)
end

"""
    parse_structure_from_mmcif_file(path::String; kwargs...) -> Structure

Parse a Structure from an mmCIF file (plain or gzip/zstd compressed).
"""
function parse_structure_from_mmcif_file(path::String; kwargs...)::Structure
    text = read_compressed(path)
    return from_mmcif(text; kwargs...)
end

"""
    read_structure_release_date(mmcif_str::String) -> Union{String,Nothing}

Extract the release date from an mmCIF string without parsing the full structure.
"""
function read_structure_release_date(mmcif_str::String)::Union{String,Nothing}
    blocks = parse_cif(mmcif_str)
    isempty(blocks) && return nothing
    return _read_release_date(blocks[1])
end

"""
    read_pdb_id_from_mmcif(mmcif_str::String) -> Union{String,Nothing}

Extract the PDB entry ID from an mmCIF string.
"""
function read_pdb_id_from_mmcif(mmcif_str::String)::Union{String,Nothing}
    blocks = parse_cif(mmcif_str)
    isempty(blocks) && return nothing
    block = blocks[1]
    # Try _entry.id first, then data block name
    eid = get_scalar(block, "_entry.id", "")
    isempty(eid) || return eid
    return isempty(block.name) ? nothing : block.name
end

"""
    parse_structure_with_assemblies(mmcif_str::String; assembly_id="1", kwargs...)
        -> Structure

Parse structure and expand the specified biological assembly.
"""
function parse_structure_with_assemblies(mmcif_str::String;
    assembly_id::String="1", kwargs...)::Structure
    s = from_mmcif(mmcif_str; kwargs...)
    blocks = parse_cif(mmcif_str)
    isempty(blocks) && return s
    block = blocks[1]
    generators, operations = parse_assembly_info(block)
    isempty(generators) && return s
    return expand_assembly(s, assembly_id, generators, operations)
end
