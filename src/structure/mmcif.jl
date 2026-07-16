"""
mmcif.jl — mmCIF parser and writer for Structure data.
"""

using Printf

# ──────────────────────────────────────────────────────────────────────────────
# mmCIF → Structure parser
# ──────────────────────────────────────────────────────────────────────────────

"""
    from_mmcif(mmcif_str::String;
               fix_mse_residues=true, fix_arginines=true,
               include_bonds=true, include_water=true,
               include_other=true) -> Structure

Parse an mmCIF string into a Structure.
Handles alternate locations (takes first), optionally fixes MSE→MET and ARG atoms.
"""
function from_mmcif(mmcif_str::String;
    fix_mse_residues::Bool = true,
    fix_arginines::Bool    = true,
    include_bonds::Bool    = true,
    include_water::Bool    = true,
    include_other::Bool    = true,
)::Structure
    blocks = parse_cif(mmcif_str)
    isempty(blocks) && return Structure(name="unknown")

    block = blocks[1]
    struct_name = block.name

    # Read release date
    release_date = _read_release_date(block)

    # Build entity type map: entity_id → entity_type (poly type or "non-polymer")
    entity_type_map = _build_entity_type_map(block)

    # Build asym→entity map: asym_id → entity_id
    asym_entity_map = _build_asym_entity_map(block)

    # Parse _atom_site loop
    atom_name    = get_loop_col(block, "_atom_site.label_atom_id")
    group_pdb    = get_loop_col(block, "_atom_site.group_pdb")
    isempty(group_pdb) && (group_pdb = get_loop_col(block, "_atom_site.group_pDB"))
    type_sym     = get_loop_col(block, "_atom_site.type_symbol")
    res_names_raw= get_loop_col(block, "_atom_site.label_comp_id")
    chain_ids    = get_loop_col(block, "_atom_site.label_asym_id")
    auth_chain   = get_loop_col(block, "_atom_site.auth_asym_id")
    seq_ids      = get_loop_col(block, "_atom_site.label_seq_id")
    auth_seq_ids = get_loop_col(block, "_atom_site.auth_seq_id")
    x_coords     = get_loop_col(block, "_atom_site.cartn_x")
    y_coords     = get_loop_col(block, "_atom_site.cartn_y")
    z_coords     = get_loop_col(block, "_atom_site.cartn_z")
    b_factors    = get_loop_col(block, "_atom_site.b_iso_or_equiv")
    occupancies  = get_loop_col(block, "_atom_site.occupancy")
    alt_locs     = get_loop_col(block, "_atom_site.label_alt_id")
    model_nums   = get_loop_col(block, "_atom_site.pdbx_pdb_model_num")

    n = length(atom_name)
    n == 0 && return Structure(name=struct_name, release_date=release_date)

    # Arrays to build
    out_atom_name    = String[]
    out_atom_element = String[]
    out_res_name     = String[]
    out_res_id       = Int[]
    out_chain_id     = String[]
    out_chain_type   = String[]
    out_x            = Float32[]
    out_y            = Float32[]
    out_z            = Float32[]
    out_b            = Float32[]
    out_occ          = Float32[]

    # Track alt_loc: skip non-first alt locs
    seen_alt = Dict{Tuple{String,String,Int,String}, String}()  # (chain,res,seq_id,atom) → first alt_loc

    for i in 1:n
        # Only take first model
        if !isempty(model_nums)
            model_nums[i] != "1" && model_nums[i] != "" && continue
        end

        # Alt location handling: skip non-first
        alt = isempty(alt_locs) ? "" : alt_locs[i]
        if alt != "" && alt != "." && alt != "?"
            aid = auth_seq_ids[i]
            seq_i = _parse_seq_id(isempty(seq_ids) ? aid : seq_ids[i])
            key = (chain_ids[i], res_names_raw[i], seq_i, atom_name[i])
            if haskey(seen_alt, key)
                continue  # skip non-first alt loc
            else
                seen_alt[key] = alt
            end
        end

        asym = chain_ids[i]
        entity_id = get(asym_entity_map, asym, "")
        chain_type = get(entity_type_map, entity_id, NON_POLYMER)

        # Filter entity types
        if chain_type in WATER_CHAIN_TYPES && !include_water
            continue
        end
        if chain_type == OTHER_CHAIN && !include_other
            continue
        end

        res_n = res_names_raw[i]
        seq_i = _parse_seq_id(isempty(seq_ids) ? (isempty(auth_seq_ids) ? "0" : auth_seq_ids[i]) : seq_ids[i])

        # For non-polymers, use auth_seq_id for residue number
        if chain_type in NON_POLYMER_CHAIN_TYPES || chain_type in WATER_CHAIN_TYPES
            seq_i = _parse_seq_id(isempty(auth_seq_ids) ? string(i) : auth_seq_ids[i])
        end

        # Fix MSE
        if fix_mse_residues && res_n == "MSE"
            res_n = "MET"
            aname = atom_name[i]
            if aname == "SE"
                continue  # Skip selenium; or replace
            end
        end

        # Fix arginine atom names (NE1→NE etc.)
        aname = atom_name[i]
        if fix_arginines && res_n == "ARG"
            aname = _fix_arg_atom(aname)
        end

        push!(out_atom_name,    _strip_quotes(aname))
        push!(out_atom_element, isempty(type_sym) ? "" : _strip_quotes(type_sym[i]))
        push!(out_res_name,     res_n)
        push!(out_res_id,       seq_i)
        push!(out_chain_id,     _strip_quotes(asym))
        push!(out_chain_type,   chain_type)
        push!(out_x,            isempty(x_coords) ? 0f0 : _parse_f32(x_coords[i]))
        push!(out_y,            isempty(y_coords) ? 0f0 : _parse_f32(y_coords[i]))
        push!(out_z,            isempty(z_coords) ? 0f0 : _parse_f32(z_coords[i]))
        push!(out_b,            isempty(b_factors) ? 0f0 : _parse_f32(b_factors[i]))
        push!(out_occ,          isempty(occupancies) ? 1f0 : _parse_f32(occupancies[i]))
    end

    # Build all_residues from _pdbx_poly_seq_scheme if available
    all_residues = _build_all_residues(block, entity_type_map, asym_entity_map)

    return Structure(
        atom_name    = out_atom_name,
        atom_element = out_atom_element,
        res_name     = out_res_name,
        res_id       = out_res_id,
        chain_id     = out_chain_id,
        chain_type   = out_chain_type,
        atom_x       = out_x,
        atom_y       = out_y,
        atom_z       = out_z,
        atom_b_factor   = out_b,
        atom_occupancy  = out_occ,
        name         = struct_name,
        release_date = release_date,
        all_residues = all_residues,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Structure → mmCIF writer
# ──────────────────────────────────────────────────────────────────────────────

"""
    to_mmcif(s::Structure) -> String

Serialize a Structure to mmCIF format string.
"""
function to_mmcif(s::Structure)::String
    buf = IOBuffer()
    _write_mmcif(buf, s)
    return String(take!(buf))
end

function _write_mmcif(io::IO, s::Structure)
    name = isempty(s.name) ? "unknown" : s.name
    println(io, "data_$name")
    println(io, "#")

    # _entry.id
    println(io, "_entry.id $(name)")
    println(io, "#")

    # Write release date if available
    if s.release_date !== nothing
        println(io, "_pdbx_audit_revision_history.revision_date  $(s.release_date)")
        println(io, "#")
    end

    # _atom_site loop
    println(io, "loop_")
    for tag in ["_atom_site.group_PDB", "_atom_site.id",
                "_atom_site.type_symbol", "_atom_site.label_atom_id",
                "_atom_site.label_comp_id", "_atom_site.label_asym_id",
                "_atom_site.label_seq_id", "_atom_site.Cartn_x",
                "_atom_site.Cartn_y", "_atom_site.Cartn_z",
                "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_PDB_model_num"]
        println(io, tag)
    end

    for i in 1:length(s)
        group = is_hetatm(s.res_name[i], s.chain_type[i]) ? "HETATM" : "ATOM"
        @printf(io, "%-6s %5d %-4s %-4s %-3s %2s %5d %8.3f %8.3f %8.3f %6.2f %6.2f 1\n",
            group, i,
            isempty(s.atom_element[i]) ? "." : s.atom_element[i],
            s.atom_name[i], s.res_name[i], s.chain_id[i], s.res_id[i],
            s.atom_x[i], s.atom_y[i], s.atom_z[i],
            s.atom_occupancy[i], s.atom_b_factor[i])
    end
    println(io, "#")
end

"""
    to_mmcif_dict(s::Structure) -> Dict{String,Any}

Serialize Structure to mmCIF dictionary representation.
"""
function to_mmcif_dict(s::Structure)::Dict{String,Any}
    n = length(s)
    return Dict{String,Any}(
        "_atom_site.group_PDB"  => [is_hetatm(s.res_name[i], s.chain_type[i]) ? "HETATM" : "ATOM" for i in 1:n],
        "_atom_site.id"         => string.(1:n),
        "_atom_site.type_symbol"     => s.atom_element,
        "_atom_site.label_atom_id"   => s.atom_name,
        "_atom_site.label_comp_id"   => s.res_name,
        "_atom_site.label_asym_id"   => s.chain_id,
        "_atom_site.label_seq_id"    => string.(s.res_id),
        "_atom_site.Cartn_x"         => [@sprintf("%.3f", x) for x in s.atom_x],
        "_atom_site.Cartn_y"         => [@sprintf("%.3f", y) for y in s.atom_y],
        "_atom_site.Cartn_z"         => [@sprintf("%.3f", z) for z in s.atom_z],
        "_atom_site.occupancy"       => string.(s.atom_occupancy),
        "_atom_site.B_iso_or_equiv"  => string.(s.atom_b_factor),
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# HETATM classification
# ──────────────────────────────────────────────────────────────────────────────

"""
    is_hetatm(res_name::String, chain_type::String) -> Bool

Return true if this atom should be written as HETATM in PDB/mmCIF format.
"""
function is_hetatm(res_name::String, chain_type::String)::Bool
    # Standard polymer residues are ATOM records
    if chain_type == PROTEIN_CHAIN
        return res_name ∉ PROTEIN_TYPES && res_name != UNK && res_name != MSE
    elseif chain_type == RNA_CHAIN
        return res_name ∉ RNA_TYPES && res_name != UNK_RNA
    elseif chain_type == DNA_CHAIN
        return res_name ∉ DNA_TYPES && res_name != UNK_DNA
    else
        return true  # NON_POLYMER, WATER, etc.
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

function _parse_seq_id(s::String)::Int
    s = strip(s)
    (isempty(s) || s == "." || s == "?") && return 0
    # Handle insertion codes like "12A"
    m = match(r"^(-?\d+)", s)
    m === nothing && return 0
    return parse(Int, m.captures[1])
end

function _parse_f32(s::String)::Float32
    try; return parse(Float32, s); catch; return 0f0; end
end

function _strip_quotes(s::String)::String
    s = strip(s)
    (startswith(s,"'") && endswith(s,"'")) && return s[2:end-1]
    (startswith(s,"\"") && endswith(s,"\"")) && return s[2:end-1]
    return s
end

function _fix_arg_atom(name::String)::String
    # Some PDB files use non-standard arginine atom names
    mapping = Dict("NE1"=>"NE","NH11"=>"NH1","NH12"=>"NH1","NH21"=>"NH2","NH22"=>"NH2")
    return get(mapping, name, name)
end

function _build_entity_type_map(block::CifDict)::Dict{String,String}
    result = Dict{String,String}()
    # Polymer entities
    poly_entity = get_loop_col(block, "_entity_poly.entity_id")
    poly_type   = get_loop_col(block, "_entity_poly.type")
    for i in 1:length(poly_entity)
        result[poly_entity[i]] = get(poly_type, i, OTHER_CHAIN)
    end
    # Non-polymer entities
    np_entity = get_loop_col(block, "_pdbx_entity_nonpoly.entity_id")
    np_comp   = get_loop_col(block, "_pdbx_entity_nonpoly.comp_id")
    for i in 1:length(np_entity)
        comp = get(np_comp, i, "UNL")
        result[np_entity[i]] = comp in WATER_COMPONENT_IDS ? WATER : NON_POLYMER
    end
    # Branched (glycan) entities
    br_entity = get_loop_col(block, "_pdbx_entity_branch.entity_id")
    for eid in br_entity
        result[eid] = BRANCHED
    end
    # Fallback: entity table
    ent_ids  = get_loop_col(block, "_entity.id")
    ent_type = get_loop_col(block, "_entity.type")
    for i in 1:length(ent_ids)
        eid = ent_ids[i]
        haskey(result, eid) && continue
        et = get(ent_type, i, "non-polymer")
        if et == "water"
            result[eid] = WATER
        elseif et == "polymer"
            result[eid] = OTHER_CHAIN
        else
            result[eid] = NON_POLYMER
        end
    end
    return result
end

function _build_asym_entity_map(block::CifDict)::Dict{String,String}
    result = Dict{String,String}()
    asym_ids   = get_loop_col(block, "_struct_asym.id")
    entity_ids = get_loop_col(block, "_struct_asym.entity_id")
    for i in 1:length(asym_ids)
        result[asym_ids[i]] = get(entity_ids, i, "")
    end
    return result
end

function _read_release_date(block::CifDict)::Union{String,Nothing}
    dates = get_loop_col(block, "_pdbx_audit_revision_history.revision_date")
    isempty(dates) || return dates[1]
    d = get_scalar(block, "_database_2.database_code", "")
    isempty(d) || return nothing
    return nothing
end

function _build_all_residues(block::CifDict,
    entity_type_map::Dict{String,String},
    asym_entity_map::Dict{String,String},
)::Dict{String,Vector{Tuple{String,Int}}}
    result = Dict{String,Vector{Tuple{String,Int}}}()
    # Use _pdbx_poly_seq_scheme for complete residue list
    asym_ids = get_loop_col(block, "_pdbx_poly_seq_scheme.asym_id")
    seq_ids  = get_loop_col(block, "_pdbx_poly_seq_scheme.seq_id")
    mon_ids  = get_loop_col(block, "_pdbx_poly_seq_scheme.mon_id")
    for i in 1:length(asym_ids)
        cid = asym_ids[i]
        rid = _parse_seq_id(seq_ids[i])
        rn  = mon_ids[i]
        if !haskey(result, cid)
            result[cid] = Tuple{String,Int}[]
        end
        push!(result[cid], (rn, rid))
    end
    return result
end
