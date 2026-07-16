"""
structure_stores.jl — Structure database lookup for template featurization.
"""

using Logging

# ──────────────────────────────────────────────────────────────────────────────
# PDB structure store
# ──────────────────────────────────────────────────────────────────────────────

"""
    PdbStructureStore

Looks up PDB structures from a directory of mmCIF files.
"""
struct PdbStructureStore
    pdb_dir::String
end

function Base.show(io::IO, s::PdbStructureStore)
    print(io, "PdbStructureStore($(s.pdb_dir))")
end

"""
    get_structure(store::PdbStructureStore, pdb_id::String) -> Union{Structure,Nothing}

Load a PDB structure by ID. Returns nothing if not found.
Searches for files named:
  - {pdb_id}.cif, {pdb_id}.cif.gz, {pdb_id}.bcif.gz
  - Subdirectory layout: {pdb_id[2:3]}/{pdb_id}.cif, etc.
"""
function get_structure(store::PdbStructureStore, pdb_id::String)::Union{Structure,Nothing}
    pid = lowercase(strip(pdb_id))
    # Remove chain spec if present (e.g., "4HHB_A" → "4hhb")
    pid = split(pid, "_")[1]

    candidates = [
        joinpath(store.pdb_dir, "$pid.cif"),
        joinpath(store.pdb_dir, "$pid.cif.gz"),
        joinpath(store.pdb_dir, "$(pid[3:4])", "$pid.cif"),
        joinpath(store.pdb_dir, "$(pid[3:4])", "$pid.cif.gz"),
        joinpath(store.pdb_dir, "$pid.bcif"),
    ]

    for path in candidates
        isfile(path) || continue
        try
            return parse_structure_from_mmcif_file(path)
        catch e
            @warn "Failed to load structure $pid from $path: $e"
        end
    end

    @debug "Structure $pid not found in $(store.pdb_dir)"
    return nothing
end

# ──────────────────────────────────────────────────────────────────────────────
# SEQRES database
# ──────────────────────────────────────────────────────────────────────────────

"""
    SeqresDatabase

Looks up PDB sequences from a FASTA file of PDB SEQRES sequences.
"""
struct SeqresDatabase
    path::String
    _index::Dict{String,Tuple{Int,Int}}  # accession → (offset, length) in file
end

function SeqresDatabase(path::String)
    idx = Dict{String,Tuple{Int,Int}}()
    if isfile(path)
        # Build offset index lazily (scan file once)
        offset = 0
        open(path, "r") do io
            while !eof(io)
                line = readline(io)
                len  = length(line) + 1  # +1 for newline
                if startswith(line, '>')
                    desc = strip(line[2:end])
                    acc  = split(desc)[1]
                    idx[acc] = (offset, 0)  # simplified: store offset
                end
                offset += len
            end
        end
    end
    return SeqresDatabase(path, idx)
end

function Base.show(io::IO, s::SeqresDatabase)
    print(io, "SeqresDatabase($(s.path), $(length(s._index)) entries)")
end

"""
    get_sequence(db::SeqresDatabase, accession::String) -> Union{String,Nothing}

Look up a sequence by accession.
"""
function get_sequence(db::SeqresDatabase, accession::String)::Union{String,Nothing}
    isfile(db.path) || return nothing
    # Simple linear scan (for production, use an index)
    for record in parse_fasta_file(db.path)
        if startswith(record.description, accession)
            return record.sequence
        end
    end
    return nothing
end
