"""
mmcif_metadata.jl — Utilities to extract metadata from mmCIF structures.
"""

using Dates
using Logging

"""
    extract_release_date(mmcif_dict::CifDict) -> Union{Date,Nothing}

Extract the deposition or release date from an mmCIF dictionary.
"""
function extract_release_date(mmcif_dict::CifDict)::Union{Date,Nothing}
    # Try _pdbx_audit_revision_history.revision_date
    rev_dates = get_loop_col(mmcif_dict, "_pdbx_audit_revision_history", "revision_date")
    if !isempty(rev_dates)
        d = _parse_date(rev_dates[1])
        d !== nothing && return d
    end

    # Try _struct.pdbx_model_details
    deposition = get_scalar(mmcif_dict, "_pdbx_database_status.recvd_initial_deposition_date")
    deposition !== nothing && return _parse_date(deposition)

    release = get_scalar(mmcif_dict, "_pdbx_deposit_group.initial_deposition_date")
    release !== nothing && return _parse_date(release)

    return nothing
end

function _parse_date(s::String)::Union{Date,Nothing}
    s = strip(s)
    # Try YYYY-MM-DD
    m = match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m !== nothing
        try
            return Date(parse(Int, m.captures[1]),
                        parse(Int, m.captures[2]),
                        parse(Int, m.captures[3]))
        catch; end
    end
    # Try YYYY/MM/DD
    m = match(r"^(\d{4})/(\d{2})/(\d{2})", s)
    if m !== nothing
        try
            return Date(parse(Int, m.captures[1]),
                        parse(Int, m.captures[2]),
                        parse(Int, m.captures[3]))
        catch; end
    end
    return nothing
end

"""
    extract_entity_chain_map(mmcif_dict::CifDict) -> Dict{String,String}

Extract a mapping from chain ID to entity type string.
"""
function extract_entity_chain_map(mmcif_dict::CifDict)::Dict{String,String}
    mapping = Dict{String,String}()
    chain_ids = get_loop_col(mmcif_dict, "_struct_asym", "id")
    entity_ids_loop = get_loop_col(mmcif_dict, "_struct_asym", "entity_id")
    isempty(chain_ids) && return mapping

    # Build entity_id → entity_type map
    ent_ids   = get_loop_col(mmcif_dict, "_entity", "id")
    ent_types = get_loop_col(mmcif_dict, "_entity", "type")
    entity_type_map = Dict(zip(ent_ids, ent_types))

    for (cid, eid) in zip(chain_ids, entity_ids_loop)
        etype = get(entity_type_map, eid, "other")
        mapping[cid] = etype
    end
    return mapping
end

"""
    extract_resolution(mmcif_dict::CifDict) -> Union{Float32,Nothing}

Extract the structure resolution from an mmCIF dictionary.
"""
function extract_resolution(mmcif_dict::CifDict)::Union{Float32,Nothing}
    # X-ray
    res = get_scalar(mmcif_dict, "_refine.ls_d_res_high")
    res !== nothing && return _parse_f32(res)

    # Electron microscopy
    em_res = get_scalar(mmcif_dict, "_em_3d_reconstruction.resolution")
    em_res !== nothing && return _parse_f32(em_res)

    # NMR
    get_scalar(mmcif_dict, "_struct.pdbx_method_details") !== nothing && return nothing

    return nothing
end

"""
    extract_pdb_id(mmcif_dict::CifDict) -> Union{String,Nothing}

Extract the PDB ID from an mmCIF dictionary.
"""
function extract_pdb_id(mmcif_dict::CifDict)::Union{String,Nothing}
    entry_id = get_scalar(mmcif_dict, "_entry.id")
    entry_id !== nothing && return lowercase(strip(entry_id))
    return nothing
end

"""
    extract_method(mmcif_dict::CifDict) -> String

Extract the experimental method (X-RAY, NMR, EM, etc.).
"""
function extract_method(mmcif_dict::CifDict)::String
    methods = get_loop_col(mmcif_dict, "_exptl", "method")
    isempty(methods) && return "UNKNOWN"
    return uppercase(join(strip.(methods), ","))
end

"""
    extract_chain_entity_type_map(mmcif_dict::CifDict) -> Dict{String,String}

Return mapping from asym_id (chain) to entity_type (polymer, non-polymer, water).
"""
function extract_chain_entity_type_map(mmcif_dict::CifDict)::Dict{String,String}
    return extract_entity_chain_map(mmcif_dict)
end
