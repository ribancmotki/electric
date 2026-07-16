"""
Safe serialisation/deserialisation utilities (Julia equivalent of Python's safe_pickle).
Uses Julia's built-in Serialization module.
"""

using Serialization
using SHA
using CodecZstd

"""
    safe_save(path::String, data)

Serialise data to path using Julia serialisation. Atomically writes via a
temporary file to avoid partial writes on failure.
"""
function safe_save(path::String, data)
    dir = dirname(path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    tmp_path = path * ".tmp_" * string(rand(UInt32), base=16)
    try
        open(tmp_path, "w") do io
            serialize(io, data)
        end
        mv(tmp_path, path; force=true)
    catch e
        isfile(tmp_path) && rm(tmp_path; force=true)
        rethrow(e)
    end
    return path
end

"""
    safe_load(path::String)

Deserialise data from path. Raises an error if the file does not exist or
cannot be deserialised.
"""
function safe_load(path::String)
    if !isfile(path)
        error("Serialised file not found: $path")
    end
    return open(deserialize, path)
end

"""
    safe_save_compressed(path::String, data)

Serialise data and compress with zstandard. Path should end in .zst.
"""
function safe_save_compressed(path::String, data)
    dir = dirname(path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    tmp_path = path * ".tmp_" * string(rand(UInt32), base=16)
    try
        open(tmp_path, "w") do io
            stream = ZstdCompressorStream(io)
            serialize(stream, data)
            close(stream)
        end
        mv(tmp_path, path; force=true)
    catch e
        isfile(tmp_path) && rm(tmp_path; force=true)
        rethrow(e)
    end
    return path
end

"""
    safe_load_compressed(path::String)

Decompress with zstandard and deserialise data.
"""
function safe_load_compressed(path::String)
    if !isfile(path)
        error("Compressed serialised file not found: $path")
    end
    return open(path) do io
        stream = ZstdDecompressorStream(io)
        data = deserialize(stream)
        close(stream)
        data
    end
end

"""
    sha256_of_array(arr::AbstractArray) -> String

Compute the SHA-256 hash of the raw bytes of an array and return as hex string.
Used for golden test comparison.
"""
function sha256_of_array(arr::AbstractArray)::String
    bytes_data = reinterpret(UInt8, collect(vec(arr)))
    return bytes2hex(sha256(bytes_data))
end

"""
    sha256_of_string(s::String) -> String

Compute SHA-256 hash of a UTF-8 string and return as hex string.
"""
function sha256_of_string(s::String)::String
    return bytes2hex(sha256(Vector{UInt8}(s)))
end
