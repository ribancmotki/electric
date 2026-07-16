"""
PDB structure store for template retrieval.
"""

"""
    PdbStructureStore

Interface for retrieving PDB mmCIF files by PDB ID.
"""
struct PdbStructureStore
    mmcif_dir::String  # Directory containing {pdb_id}.cif files
end

"""
    get_structure(store::PdbStructureStore, pdb_id::String) -> Union{String,Nothing}

Retrieve the mmCIF content for the given PDB ID.
Tries various case combinations for the filename.
Returns nothing if not found.
"""
function get_structure(store::PdbStructureStore, pdb_id::String)::Union{String,Nothing}
    pdb_id_lower = lowercase(pdb_id)
    pdb_id_upper = uppercase(pdb_id)

    # Try different filename formats
    candidates = [
        joinpath(store.mmcif_dir, "$(pdb_id_lower).cif"),
        joinpath(store.mmcif_dir, "$(pdb_id_upper).cif"),
        joinpath(store.mmcif_dir, "$pdb_id.cif"),
        # Try subdirectory format: mmcif_files/ab/1abc.cif
        joinpath(store.mmcif_dir, pdb_id_lower[2:3], "$(pdb_id_lower).cif"),
    ]

    for path in candidates
        if isfile(path)
            return read(path, String)
        end
    end
    @debug "PDB structure not found for $pdb_id in $(store.mmcif_dir)"
    return nothing
end

"""
    get_structure!(store::PdbStructureStore, pdb_id::String) -> String

Like get_structure but raises an error if not found.
"""
function get_structure!(store::PdbStructureStore, pdb_id::String)::String
    result = get_structure(store, pdb_id)
    result === nothing && error("PDB structure '$pdb_id' not found in $(store.mmcif_dir)")
    return result
end

"""
    list_available_pdb_ids(store::PdbStructureStore) -> Vector{String}

Return a list of all PDB IDs available in the store.
"""
function list_available_pdb_ids(store::PdbStructureStore)::Vector{String}
    isdir(store.mmcif_dir) || return String[]
    ids = String[]
    for f in readdir(store.mmcif_dir; join=false)
        if endswith(f, ".cif")
            push!(ids, f[1:end-4])
        end
    end
    return ids
end

"""
    SeqresDatabase

Holds the PDB seqres FASTA for template searching.
"""
struct SeqresDatabase
    fasta_path::String
end

"""
    read_seqres(db::SeqresDatabase) -> Vector{Tuple{String,String}}

Read all seqres sequences from the FASTA database.
"""
function read_seqres(db::SeqresDatabase)::Vector{Tuple{String,String}}
    isfile(db.fasta_path) || error("Seqres FASTA not found: $(db.fasta_path)")
    return parse_fasta(read(db.fasta_path, String))
end
