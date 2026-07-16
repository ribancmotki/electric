"""
Top-level parsing entry points for structure files.
"""

"""
    parse_structure_from_mmcif_string(mmcif_str::String) -> Structure

Parse an mmCIF-format string into a Structure object.
This is the primary entry point for loading structures.
"""
function parse_structure_from_mmcif_string(mmcif_str::String)::Structure
    mmcif_dict = parse_mmcif(mmcif_str)
    return mmcif_to_structure(mmcif_dict)
end

"""
    parse_structure_from_mmcif_file(path::String) -> Structure

Parse an mmCIF file at the given path into a Structure object.
"""
function parse_structure_from_mmcif_file(path::String)::Structure
    isfile(path) || error("mmCIF file not found: $path")
    mmcif_str = read(path, String)
    return parse_structure_from_mmcif_string(mmcif_str)
end

"""
    read_structure_release_date(mmcif_str::String) -> Union{Date,Nothing}

Extract the release date from an mmCIF structure string.
Uses _pdbx_audit_revision_history.revision_date for the first (lowest ordinal) entry.
"""
function read_structure_release_date(mmcif_str::String)::Union{Date,Nothing}
    mmcif_dict = parse_mmcif(mmcif_str)
    isempty(mmcif_dict) && return nothing
    block = first(values(mmcif_dict))
    dates = _get_loop_col(block, "_pdbx_audit_revision_history.revision_date", String[])
    isempty(dates) && return nothing
    for ds in dates
        (isempty(ds) || ds == "?" || ds == ".") && continue
        try
            return Date(ds, dateformat"yyyy-mm-dd")
        catch
            # Try alternative format
            try
                return Date(ds[1:10])
            catch
                continue
            end
        end
    end
    return nothing
end

"""
    read_pdb_id_from_mmcif(mmcif_str::String) -> Union{String,Nothing}

Extract the PDB ID from an mmCIF string.
"""
function read_pdb_id_from_mmcif(mmcif_str::String)::Union{String,Nothing}
    mmcif_dict = parse_mmcif(mmcif_str)
    isempty(mmcif_dict) && return nothing
    block = first(values(mmcif_dict))
    entry_id = get(block, "_entry.id", nothing)
    entry_id !== nothing && return String(entry_id)
    # Fallback to data block name
    block_name = first(keys(mmcif_dict))
    return block_name
end
