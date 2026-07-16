"""
Resource path resolution utilities.
"""

"""
    get_package_data_dir() -> String

Return the path to the package data directory (where compiled CCD database lives).
"""
function get_package_data_dir()::String
    pkg_dir = joinpath(@__DIR__, "..", "..", "data")
    if !isdir(pkg_dir)
        mkpath(pkg_dir)
    end
    return pkg_dir
end

"""
    get_ccd_database_path() -> String

Return the path to the compiled CCD binary database.
"""
function get_ccd_database_path()::String
    return joinpath(get_package_data_dir(), "ccd_database.jld2")
end

"""
    get_test_data_dir() -> String

Return the path to the test data directory.
"""
function get_test_data_dir()::String
    return joinpath(@__DIR__, "..", "..", "test_data")
end

"""
    require_file(path::String, description::String)

Raise an error if path does not exist.
"""
function require_file(path::String, description::String)
    if !isfile(path)
        error("Required $description not found at: $path")
    end
end

"""
    require_dir(path::String, description::String)

Raise an error if path does not exist as a directory.
"""
function require_dir(path::String, description::String)
    if !isdir(path)
        error("Required $description directory not found at: $path")
    end
end

"""
    replace_db_dir(path::String, db_dirs::Vector{String}) -> String

Substitute \${DB_DIR} in path with the first db_dir where the resolved path exists.
Raises SystemError if no valid substitution is found.
"""
function replace_db_dir(path::String, db_dirs::Vector{String})::String
    if !occursin("\${DB_DIR}", path)
        return path
    end
    for db_dir in db_dirs
        candidate = replace(path, "\${DB_DIR}" => db_dir)
        # For sharded paths, check the directory or first shard
        shard_paths = get_sharded_paths(candidate)
        if shard_paths !== nothing
            if !isempty(shard_paths) && isfile(first(shard_paths))
                return candidate
            end
        elseif isfile(candidate) || isdir(candidate)
            return candidate
        end
    end
    throw(SystemError("No valid db_dir substitution found for path: $path (tried: $(join(db_dirs, ", ")))", 2))
end
