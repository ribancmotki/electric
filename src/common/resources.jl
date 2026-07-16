"""
resources.jl — Resource path utilities for databases and package data.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Package data directory
# ──────────────────────────────────────────────────────────────────────────────

"""
    get_package_data_dir() -> String

Return the path to the package data directory, checking several common locations.
Raises an error if none can be found.
"""
function get_package_data_dir()::String
    # 1. Environment variable override
    if haskey(ENV, "STRUCT_PRED_DATA_DIR")
        d = ENV["STRUCT_PRED_DATA_DIR"]
        isdir(d) && return d
    end
    # 2. Adjacent to module source
    candidates = [
        joinpath(@__DIR__, "..", "..", "data"),
        joinpath(@__DIR__, "..", "..", "..", "data"),
        "/data",
        "/databases",
    ]
    for c in candidates
        isdir(c) && return abspath(c)
    end
    error("Cannot find package data directory. Set STRUCT_PRED_DATA_DIR environment variable.")
end

"""
    get_ccd_database_path() -> String

Return the path to the CCD binary database.
"""
function get_ccd_database_path()::String
    data_dir = get_package_data_dir()
    candidates = [
        joinpath(data_dir, "ccd.bin"),
        joinpath(data_dir, "ccd", "ccd.bin"),
        get(ENV, "CCD_PATH", ""),
    ]
    for p in candidates
        !isempty(p) && isfile(p) && return p
    end
    @warn "CCD database not found. Using empty CCD."
    return ""
end

# ──────────────────────────────────────────────────────────────────────────────
# Database path helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    replace_db_dir(path::String, db_dir::String) -> String

Replace the database directory prefix in a path.
If the path starts with a known database root, replace it with `db_dir`.
"""
function replace_db_dir(path::String, db_dir::String)::String
    isempty(path) && return path
    expanded = expand_env_vars(path)
    # If path is absolute and exists under db_dir, use that
    basename_path = basename(expanded)
    candidate = joinpath(db_dir, basename_path)
    isfile(candidate) && return candidate
    # Try sub-directory
    rel = _try_relativize(expanded, db_dir)
    rel !== nothing && return joinpath(db_dir, rel)
    # If the path contains a shard pattern, handle shards
    if contains(expanded, "@")
        return _replace_sharded_path(expanded, db_dir)
    end
    # Return as-is if we can't figure it out
    return expanded
end

function _try_relativize(path::String, base_dir::String)::Union{String,Nothing}
    # Try to find the common tail component
    abs_path = abspath(path)
    known_db_roots = ["/databases", "/data", "/mnt"]
    for root in known_db_roots
        if startswith(abs_path, root)
            rel = abs_path[length(root)+1:end]
            return lstrip(rel, '/')
        end
    end
    return nothing
end

function _replace_sharded_path(path::String, db_dir::String)::String
    # e.g. /databases/uniref90/uniref90.fasta@4 → db_dir/uniref90/uniref90.fasta@4
    m = match(r"(.+?)@(\d+)$", path)
    m === nothing && return path
    base, n_shards = m.captures
    base_name = basename(base)
    new_base = joinpath(db_dir, base_name)
    return "$new_base@$n_shards"
end

# ──────────────────────────────────────────────────────────────────────────────
# Standard database names
# ──────────────────────────────────────────────────────────────────────────────

"""Database subdirectory names as used in the download script."""
const DB_NAMES = (
    uniref90     = "uniref90",
    mgnify       = "mgnify",
    small_bfd    = "small_bfd",
    uniprot      = "uniprot",
    ntrna        = "nt_rna",
    rfam         = "rfam",
    rnacentral   = "rnacentral",
    pdb_seqres   = "pdb_seqres",
    pdb_mmcif    = "pdb_mmcif",
)

"""
    get_database_path(db_dir::String, db_name::Symbol) -> String

Return the expected path for a named database in db_dir.
"""
function get_database_path(db_dir::String, db_name::Symbol)::String
    return joinpath(db_dir, string(DB_NAMES[db_name]))
end
