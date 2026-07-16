"""
mmCIF metadata extraction for template filtering and chain information.
"""

using Dates

"""
    extract_release_date(mmcif_dict::Dict{String,Any}) -> Union{Date,Nothing}

Extract the first release date from _pdbx_audit_revision_history.
"""
function extract_release_date(mmcif_dict::Dict{String,Any})::Union{Date,Nothing}
    isempty(mmcif_dict) && return nothing
    block = first(values(mmcif_dict))

    dates    = _get_loop_col(block, "_pdbx_audit_revision_history.revision_date", String[])
    ordinals = _get_loop_col(block, "_pdbx_audit_revision_history.ordinal",        String[])

    if isempty(dates)
        # Try _database_pdb_rev.date
        rev_dates = _get_loop_col(block, "_database_pdb_rev.date", String[])
        isempty(rev_dates) && return nothing
        dates = rev_dates
    end

    # Find the entry with the lowest ordinal (= initial deposition/release)
    target_date = nothing
    if !isempty(ordinals)
        pairs = collect(zip(ordinals, dates))
        sort!(pairs, by = p -> tryparse(Int, p[1]) !== nothing ? parse(Int, p[1]) : 999)
        target_date = last(first(pairs))
    else
        target_date = first(dates)
    end

    isempty(target_date) || target_date == "?" && return nothing
    try
        return Date(target_date[1:10])
    catch
        return nothing
    end
end

"""
    extract_entity_chain_map(mmcif_dict::Dict{String,Any}) -> Dict{String,String}

Return a mapping from asym_id (chain) to entity_id.
"""
function extract_entity_chain_map(mmcif_dict::Dict{String,Any})::Dict{String,String}
    isempty(mmcif_dict) && return Dict{String,String}()
    block = first(values(mmcif_dict))
    asym_ids   = _get_loop_col(block, "_struct_asym.id", String[])
    entity_ids = _get_loop_col(block, "_struct_asym.entity_id", String[])
    return Dict(a => e for (a, e) in zip(asym_ids, entity_ids))
end

"""
    extract_entity_sequences(mmcif_dict::Dict{String,Any}) -> Dict{String,String}

Return a mapping from entity_id to one-letter sequence from _entity_poly.
"""
function extract_entity_sequences(mmcif_dict::Dict{String,Any})::Dict{String,String}
    isempty(mmcif_dict) && return Dict{String,String}()
    block = first(values(mmcif_dict))
    entity_ids = _get_loop_col(block, "_entity_poly.entity_id",               String[])
    sequences  = _get_loop_col(block, "_entity_poly.pdbx_seq_one_letter_code", String[])
    result = Dict{String,String}()
    for (eid, seq) in zip(entity_ids, sequences)
        # Remove line breaks and whitespace
        result[eid] = replace(seq, r"\s+" => "")
    end
    return result
end

"""
    extract_entity_type(mmcif_dict::Dict{String,Any}) -> Dict{String,String}

Return a mapping from entity_id to entity type ("polymer", "non-polymer", "water").
"""
function extract_entity_type(mmcif_dict::Dict{String,Any})::Dict{String,String}
    isempty(mmcif_dict) && return Dict{String,String}()
    block = first(values(mmcif_dict))
    entity_ids = _get_loop_col(block, "_entity.id",   String[])
    types      = _get_loop_col(block, "_entity.type", String[])
    return Dict(e => t for (e, t) in zip(entity_ids, types))
end

"""
    extract_chain_sequence(mmcif_dict::Dict{String,Any}, chain_id::String) -> String

Return the sequence for a specific chain, looking up through entity_poly.
"""
function extract_chain_sequence(mmcif_dict::Dict{String,Any}, chain_id::String)::String
    chain_to_entity = extract_entity_chain_map(mmcif_dict)
    entity_to_seq   = extract_entity_sequences(mmcif_dict)
    entity_id = get(chain_to_entity, chain_id, nothing)
    entity_id === nothing && return ""
    return get(entity_to_seq, entity_id, "")
end

"""
    get_pdb_entry_id(mmcif_dict::Dict{String,Any}) -> String

Extract the PDB entry ID from the data block name or _entry.id field.
"""
function get_pdb_entry_id(mmcif_dict::Dict{String,Any})::String
    isempty(mmcif_dict) && return ""
    block_name = first(keys(mmcif_dict))
    block = first(values(mmcif_dict))
    entry_id = get(block, "_entry.id", nothing)
    return entry_id !== nothing ? String(entry_id) : block_name
end
