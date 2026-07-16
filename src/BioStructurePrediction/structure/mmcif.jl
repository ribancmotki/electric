"""
Full mmCIF parser and writer for biological structure data.
"""

using Dates

# ──────────────────────────────────────────────
#  Parser
# ──────────────────────────────────────────────

"""
    parse_mmcif(text::String) -> Dict{String,Any}

Parse an mmCIF file text into a nested dictionary.
Top-level keys are data block names; values are Dicts of category/field → value or loop table.
"""
function parse_mmcif(text::String)::Dict{String,Any}
    result = Dict{String,Any}()
    lines = split(text, '\n')
    n = length(lines)
    current_block = nothing
    current_data  = Dict{String,Any}()
    i = 1

    while i <= n
        line = rstrip(lines[i])

        # Skip blank lines and comments
        if isempty(line) || startswith(line, "#")
            i += 1
            continue
        end

        # New data block
        m = match(r"^data_(\S*)", line)
        if m !== nothing
            # Save previous block
            if current_block !== nothing
                result[current_block] = current_data
            end
            current_block = String(m.captures[1])
            current_data  = Dict{String,Any}()
            i += 1
            continue
        end

        # Loop construct
        if strip(line) == "loop_"
            i += 1
            # Collect headers
            headers = String[]
            while i <= n
                hline = strip(lines[i])
                if startswith(hline, "_")
                    push!(headers, hline)
                    i += 1
                else
                    break
                end
            end
            # Collect data rows
            rows = Vector{String}[]
            while i <= n
                dline = rstrip(lines[i])
                # Multi-line values start with ;
                if startswith(strip(dline), ";")
                    # Collect multi-line value
                    val_lines = String[]
                    i += 1
                    while i <= n && strip(lines[i]) != ";"
                        push!(val_lines, lines[i])
                        i += 1
                    end
                    i += 1  # skip closing ;
                    # Append as single token to last row or start new row
                    if isempty(rows) || length(last(rows)) == length(headers)
                        push!(rows, [join(val_lines, "\n")])
                    else
                        push!(last(rows), join(val_lines, "\n"))
                    end
                    continue
                end
                toks = tokenise_mmcif_line(strip(dline))
                if isempty(toks)
                    i += 1
                    if isempty(dline) || startswith(strip(dline), "#")
                        # Check if we're done
                        break
                    end
                    continue
                end
                # Check if first token is a new key or block — signals end of loop
                t1 = first(toks)
                if startswith(t1, "_") || startswith(t1, "data_") || t1 == "loop_"
                    break
                end
                # Distribute tokens into rows
                row_buf = isempty(rows) || length(last(rows)) == length(headers) ? String[] : pop!(rows)
                for tok in toks
                    push!(row_buf, tok)
                    if length(row_buf) == length(headers)
                        push!(rows, row_buf)
                        row_buf = String[]
                    end
                end
                !isempty(row_buf) && push!(rows, row_buf)
                i += 1
            end
            # Only store complete rows
            complete_rows = filter(r -> length(r) == length(headers), rows)
            # Store loop as a Dict of column arrays
            loop_data = Dict{String,Vector{String}}()
            for (j, hdr) in enumerate(headers)
                loop_data[hdr] = String[r[j] for r in complete_rows]
            end
            # Merge into current data
            for (k, v) in loop_data
                current_data[k] = v
            end
            continue
        end

        # Multi-line value (semicolon-delimited)
        if startswith(strip(line), ";")
            # This is a continuation of a previous key — skip
            while i <= n && strip(lines[i]) != ";"
                i += 1
            end
            i += 1
            continue
        end

        # Key-value pair
        m2 = match(r"^(_\S+)\s*(.*)", line)
        if m2 !== nothing
            key = String(m2.captures[1])
            rest = strip(String(m2.captures[2]))
            val = if isempty(rest)
                # Value on next line(s)
                i += 1
                if i <= n && startswith(strip(lines[i]), ";")
                    i += 1
                    val_lines = String[]
                    while i <= n && strip(lines[i]) != ";"
                        push!(val_lines, lines[i])
                        i += 1
                    end
                    i += 1
                    join(val_lines, "\n")
                elseif i <= n
                    v = strip(lines[i])
                    i += 1
                    tokenise_mmcif_line(v) |> first
                else
                    ""
                end
            else
                toks = tokenise_mmcif_line(rest)
                isempty(toks) ? rest : first(toks)
            end
            current_data[key] = val
            continue
        end

        i += 1
    end

    # Save last block
    if current_block !== nothing
        result[current_block] = current_data
    end

    return result
end

"""
    tokenise_mmcif_line(line::String) -> Vector{String}

Split an mmCIF data line into tokens, respecting single/double quoted strings.
"""
function tokenise_mmcif_line(line::String)::Vector{String}
    tokens = String[]
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == ' ' || c == '\t'
            i = nextind(line, i)
        elseif c == '\'' || c == '"'
            qchar = c
            i = nextind(line, i)
            start = i
            while i <= lastindex(line) && line[i] != qchar
                i = nextind(line, i)
            end
            push!(tokens, line[start:prevind(line, i)])
            i <= lastindex(line) && (i = nextind(line, i))
        elseif c == '#'
            break  # comment
        else
            start = i
            while i <= lastindex(line) && line[i] != ' ' && line[i] != '\t'
                i = nextind(line, i)
            end
            tok = line[start:prevind(line, i)]
            tok != "." && tok != "?" && push!(tokens, tok)
            (tok == "." || tok == "?") && push!(tokens, "")
        end
    end
    return tokens
end

# ──────────────────────────────────────────────
#  Writer
# ──────────────────────────────────────────────

"""
    write_mmcif(data::Dict{String,Any}) -> String

Write a Dict (as returned by parse_mmcif) back to mmCIF format.
"""
function write_mmcif(data::Dict{String,Any})::String
    buf = IOBuffer()
    for (block_name, block_data) in data
        println(buf, "data_$block_name")
        println(buf, "#")
        # Group keys by category
        cats = Dict{String,Vector{String}}()
        for key in keys(block_data)
            cat = join(split(key, ".")[1:end-1], ".")
            push!(get!(cats, cat, String[]), key)
        end
        for (cat, keys_in_cat) in sort(collect(cats))
            first_val = block_data[first(keys_in_cat)]
            if first_val isa Vector
                # Loop
                println(buf, "loop_")
                for k in keys_in_cat
                    println(buf, k)
                end
                n_rows = length(first_val)
                for i in 1:n_rows
                    row_vals = String[mmcif_format_value(block_data[k][i]) for k in keys_in_cat]
                    println(buf, join(row_vals, " "))
                end
                println(buf, "#")
            else
                # Key-value
                for k in keys_in_cat
                    v = block_data[k]
                    println(buf, "$k $(mmcif_format_value(string(v)))")
                end
                println(buf, "#")
            end
        end
    end
    return String(take!(buf))
end

"""
    mmcif_format_value(v::String) -> String

Format a value for mmCIF output, quoting if necessary.
"""
function mmcif_format_value(v::String)::String
    isempty(v) && return "."
    v == "?" && return "?"
    # Quote if contains spaces or special characters
    if occursin(r"[\s'\"#]", v)
        if !occursin('\'', v)
            return "'$v'"
        elseif !occursin('"', v)
            return "\"$v\""
        else
            return ";" * v * "\n;"
        end
    end
    return v
end

# ──────────────────────────────────────────────
#  mmCIF → Structure conversion
# ──────────────────────────────────────────────

"""
    mmcif_to_structure(mmcif_dict::Dict{String,Any}) -> Structure

Convert a parsed mmCIF dict to a Structure object.
"""
function mmcif_to_structure(mmcif_dict::Dict{String,Any})::Structure
    # Find the first data block
    if isempty(mmcif_dict)
        error("Empty mmCIF dict")
    end
    block_name = first(keys(mmcif_dict))
    block = mmcif_dict[block_name]

    # Extract _atom_site columns
    group_pdb   = _get_loop_col(block, "_atom_site.group_PDB",      String[])
    atom_ids    = _get_loop_col(block, "_atom_site.id",              String[])
    type_syms   = _get_loop_col(block, "_atom_site.type_symbol",     String[])
    atom_names  = _get_loop_col(block, "_atom_site.label_atom_id",   String[])
    comp_ids    = _get_loop_col(block, "_atom_site.label_comp_id",   String[])
    chain_ids   = _get_loop_col(block, "_atom_site.label_asym_id",   String[])
    entity_ids  = _get_loop_col(block, "_atom_site.label_entity_id", String[])
    seq_ids     = _get_loop_col(block, "_atom_site.label_seq_id",    String[])
    x_coords    = _get_loop_col(block, "_atom_site.Cartn_x",         String[])
    y_coords    = _get_loop_col(block, "_atom_site.Cartn_y",         String[])
    z_coords    = _get_loop_col(block, "_atom_site.Cartn_z",         String[])
    occupancies = _get_loop_col(block, "_atom_site.occupancy",       String[])
    bfactors    = _get_loop_col(block, "_atom_site.B_iso_or_equiv",  String[])
    auth_seq    = _get_loop_col(block, "_atom_site.auth_seq_id",     String[])
    auth_chain  = _get_loop_col(block, "_atom_site.auth_asym_id",    String[])
    model_nums  = _get_loop_col(block, "_atom_site.pdbx_PDB_model_num", String[])

    n = length(atom_ids)
    if n == 0
        return Structure(block_name, StructureTable(), ChainInfo[], ResidueInfo[], Bond[])
    end

    # Only take model 1 if multiple models present
    if !isempty(model_nums)
        model1_mask = model_nums .== "1"
        if any(model1_mask)
            sel = model1_mask
            atom_ids   = atom_ids[sel];   type_syms  = type_syms[sel]
            atom_names = atom_names[sel]; comp_ids   = comp_ids[sel]
            chain_ids  = chain_ids[sel];  entity_ids = entity_ids[sel]
            seq_ids    = seq_ids[sel];    x_coords   = x_coords[sel]
            y_coords   = y_coords[sel];   z_coords   = z_coords[sel]
            occupancies = occupancies[sel]; bfactors  = bfactors[sel]
            auth_seq   = auth_seq[sel];   auth_chain = auth_chain[sel]
            model_nums = model_nums[sel]; group_pdb  = group_pdb[sel]
        end
    end
    n = length(atom_ids)

    parse_float(s) = try parse(Float32, s) catch; 0f0 end

    atoms_table = StructureTable()
    add_column!(atoms_table, :group_PDB,      group_pdb)
    add_column!(atoms_table, :id,             [tryparse(Int, s) === nothing ? 0 : parse(Int, s) for s in atom_ids])
    add_column!(atoms_table, :type_symbol,    type_syms)
    add_column!(atoms_table, :label_atom_id,  atom_names)
    add_column!(atoms_table, :label_comp_id,  comp_ids)
    add_column!(atoms_table, :label_asym_id,  chain_ids)
    add_column!(atoms_table, :label_entity_id, entity_ids)
    add_column!(atoms_table, :label_seq_id,   seq_ids)
    add_column!(atoms_table, :Cartn_x,        parse_float.(x_coords))
    add_column!(atoms_table, :Cartn_y,        parse_float.(y_coords))
    add_column!(atoms_table, :Cartn_z,        parse_float.(z_coords))
    add_column!(atoms_table, :occupancy,      parse_float.(occupancies))
    add_column!(atoms_table, :B_iso_or_equiv, parse_float.(bfactors))
    add_column!(atoms_table, :auth_seq_id,    auth_seq)
    add_column!(atoms_table, :auth_asym_id,   auth_chain)

    # Build chain and residue info
    unique_chains = unique(chain_ids)
    chain_infos = ChainInfo[]
    residue_infos = ResidueInfo[]

    for cid in unique_chains
        chain_mask = chain_ids .== cid
        chain_atom_indices = findall(chain_mask)
        push!(chain_infos, ChainInfo(cid, chain_atom_indices[1], chain_atom_indices[end]))
    end

    # Residues: group by (chain_id, seq_id, comp_id)
    seen_residues = Set{Tuple{String,String,String}}()
    for (i, (cid, sid, cmpid)) in enumerate(zip(chain_ids, seq_ids, comp_ids))
        key = (cid, sid, cmpid)
        if key ∉ seen_residues
            push!(seen_residues, key)
            push!(residue_infos, ResidueInfo(cid, sid, cmpid, i))
        end
    end

    return Structure(block_name, atoms_table, chain_infos, residue_infos, Bond[])
end

function _get_loop_col(block::Dict{String,Any}, key::String, default)
    val = get(block, key, nothing)
    val === nothing && return default
    val isa Vector && return val
    return [string(val)]
end

# ──────────────────────────────────────────────
#  Structure → mmCIF output
# ──────────────────────────────────────────────

"""
    structure_to_mmcif(s::Structure; bfactors=nothing) -> String

Convert a Structure to a valid PDBx/mmCIF string.
If bfactors is provided, it must have length num_atoms(s) and will override the stored B-factors.
"""
function structure_to_mmcif(s::Structure; bfactors::Union{Vector{Float32},Nothing}=nothing)::String
    buf = IOBuffer()
    println(buf, "data_$(isempty(s.name) ? "structure" : replace(s.name, r"[^A-Za-z0-9_]" => "_"))")
    println(buf, "#")

    atoms = s.atoms
    n = nrows(atoms)
    n == 0 && return String(take!(buf))

    # Write _atom_site loop
    println(buf, "loop_")
    for field in ATOM_SITE_FIELDS
        println(buf, "_atom_site.$field")
    end

    chain_ids   = get_column(atoms, :label_asym_id)
    comp_ids    = get_column(atoms, :label_comp_id)
    seq_ids     = get_column(atoms, :label_seq_id)
    atom_names  = get_column(atoms, :label_atom_id)
    type_syms   = get_column(atoms, :type_symbol)
    entity_ids  = has_column(atoms, :label_entity_id) ? get_column(atoms, :label_entity_id) : fill("1", n)
    xs          = get_column(atoms, :Cartn_x)
    ys          = get_column(atoms, :Cartn_y)
    zs          = get_column(atoms, :Cartn_z)
    occs        = has_column(atoms, :occupancy)      ? get_column(atoms, :occupancy)      : fill(1f0, n)
    bfs         = has_column(atoms, :B_iso_or_equiv) ? get_column(atoms, :B_iso_or_equiv) : fill(0f0, n)
    auth_seq    = has_column(atoms, :auth_seq_id)    ? get_column(atoms, :auth_seq_id)    : seq_ids
    auth_chain  = has_column(atoms, :auth_asym_id)   ? get_column(atoms, :auth_asym_id)   : chain_ids
    groups      = has_column(atoms, :group_PDB)      ? get_column(atoms, :group_PDB)      :
        [is_hetatm(comp_ids[i]) ? "HETATM" : "ATOM  " for i in 1:n]

    for i in 1:n
        bf = bfactors !== nothing ? bfactors[i] : bfs[i]
        row = join([
            mmcif_format_value(groups[i]),
            string(i),
            mmcif_format_value(type_syms[i]),
            mmcif_format_value(atom_names[i]),
            ".",                                     # alt_id
            mmcif_format_value(comp_ids[i]),
            mmcif_format_value(chain_ids[i]),
            mmcif_format_value(string(entity_ids[i])),
            mmcif_format_value(string(seq_ids[i])),
            "?",                                     # ins_code
            @sprintf("%.3f", xs[i]),
            @sprintf("%.3f", ys[i]),
            @sprintf("%.3f", zs[i]),
            @sprintf("%.2f", occs[i]),
            @sprintf("%.2f", bf),
            mmcif_format_value(string(auth_seq[i])),
            mmcif_format_value(string(auth_chain[i])),
            "1",                                     # model_num
        ], " ")
        println(buf, row)
    end
    println(buf, "#")

    return String(take!(buf))
end

"""
    is_hetatm(comp_id::String) -> Bool

Return true if the component should be written as HETATM rather than ATOM.
"""
function is_hetatm(comp_id::String)::Bool
    return !(is_standard_amino_acid(comp_id) || is_rna_residue(comp_id) || is_dna_residue(comp_id))
end
