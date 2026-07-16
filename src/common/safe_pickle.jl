"""
safe_pickle.jl — Safe serialization/deserialization with checksums and compression.
"""

using Serialization
using SHA
using Logging

# Optional CodecZstd
const _HAS_ZSTD = try
    @eval using CodecZstd
    true
catch
    false
end

# ──────────────────────────────────────────────────────────────────────────────
# Core save / load
# ──────────────────────────────────────────────────────────────────────────────

"""
    safe_save(path::String, data; compress=false)

Serialize `data` to `path` using Julia's Serialization. Optionally zstd-compress.
Writes an accompanying `.sha256` checksum file.
"""
function safe_save(path::String, data; compress::Bool=false)
    mkpath(dirname(abspath(path)))
    tmp = path * ".tmp"
    try
        if compress && _HAS_ZSTD
            open(tmp, "w") do raw
                cs = CodecZstd.ZstdCompressorStream(raw)
                serialize(cs, data)
                close(cs)
            end
        else
            open(tmp, "w") do io
                serialize(io, data)
            end
        end
        mv(tmp, path; force=true)
        # Write checksum
        sha = sha256_of_file(path)
        write(path * ".sha256", bytes2hex(sha))
    catch e
        isfile(tmp) && rm(tmp, force=true)
        rethrow(e)
    end
end

"""
    safe_load(path::String; compress=false, verify_checksum=false)

Deserialize an object from `path`. Optionally decompress and verify checksum.
"""
function safe_load(path::String; compress::Bool=false, verify_checksum::Bool=false)
    if verify_checksum
        sha_path = path * ".sha256"
        if isfile(sha_path)
            expected = strip(read(sha_path, String))
            actual   = bytes2hex(sha256_of_file(path))
            @assert actual == expected "Checksum mismatch for $path"
        else
            @warn "No checksum file found for $path"
        end
    end
    if compress && _HAS_ZSTD
        return open(path, "r") do raw
            cs = CodecZstd.ZstdDecompressorStream(raw)
            data = deserialize(cs)
            close(cs)
            data
        end
    else
        return open(path, "r") do io
            deserialize(io)
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Zstd-specific variants
# ──────────────────────────────────────────────────────────────────────────────

"""
    safe_save_zstd(path::String, data)

Save with zstd compression. Falls back to uncompressed if CodecZstd not available.
"""
function safe_save_zstd(path::String, data)
    safe_save(path, data; compress=_HAS_ZSTD)
end

"""
    safe_load_zstd(path::String)

Load with zstd decompression. Falls back to uncompressed if CodecZstd not available.
"""
function safe_load_zstd(path::String)
    safe_load(path; compress=_HAS_ZSTD)
end

# ──────────────────────────────────────────────────────────────────────────────
# Checksum utilities
# ──────────────────────────────────────────────────────────────────────────────

"""
    sha256_of_file(path::String) -> Vector{UInt8}

Compute SHA-256 digest of a file.
"""
function sha256_of_file(path::String)::Vector{UInt8}
    return open(path, "r") do io
        sha256(io)
    end
end

"""
    sha256_of_array(arr::AbstractArray) -> String

Compute SHA-256 hex digest of an array's raw bytes.
"""
function sha256_of_array(arr::AbstractArray)::String
    bytes = reinterpret(UInt8, vec(arr))
    return bytes2hex(sha256(bytes))
end

# ──────────────────────────────────────────────────────────────────────────────
# Magic-byte detection for compressed files
# ──────────────────────────────────────────────────────────────────────────────

"""
    detect_compression(path::String) -> Symbol

Detect compression format from file magic bytes.
Returns :zstd, :gzip, :xz, or :none.
"""
function detect_compression(path::String)::Symbol
    isfile(path) || return :none
    magic = open(path, "r") do io
        read(io, 6)
    end
    length(magic) >= 4 || return :none
    # Zstd magic: 0xFD2FB528
    magic[1:4] == UInt8[0xFD, 0x2F, 0xB5, 0x28] && return :zstd
    # Gzip magic: 0x1F8B
    magic[1:2] == UInt8[0x1F, 0x8B] && return :gzip
    # XZ magic: 0xFD377A585A00
    length(magic) >= 6 && magic[1:6] == UInt8[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00] && return :xz
    return :none
end

"""
    read_compressed(path::String) -> String

Read a possibly compressed file as text. Supports gzip, xz, zstd.
"""
function read_compressed(path::String)::String
    fmt = detect_compression(path)
    if fmt == :none
        return read(path, String)
    elseif fmt == :zstd && _HAS_ZSTD
        return open(path, "r") do raw
            String(read(CodecZstd.ZstdDecompressorStream(raw)))
        end
    elseif fmt == :gzip
        # Use GZip or system gunzip
        try
            @eval using GZip
            return GZip.open(path, "r") do io
                read(io, String)
            end
        catch
            data = read(`gunzip -c $path`, String)
            return data
        end
    else
        # Fallback: try reading as raw text
        return read(path, String)
    end
end
