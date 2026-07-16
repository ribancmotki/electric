"""
Sharded database support for Jackhmmer/Nhmmer parallel search.
"""

"""
    get_sharded_paths(path::String) -> Union{Vector{String},Nothing}

If path matches a shard pattern, return all shard file paths; else return nothing.

Supported patterns:
- prefix@N  — N shards at prefix-00000-of-NNNNN, ..., prefix-NNNNN-of-NNNNN
- prefix-NNNNN-of-NNNNN — a single shard path; returns all sibling shards
"""
function get_sharded_paths(path::String)::Union{Vector{String},Nothing}
    # Pattern: path@N
    m = match(r"^(.+)@(\d+)$", path)
    if m !== nothing
        prefix = String(m.captures[1])
        n      = parse(Int, m.captures[2])
        ndigits = length(string(n - 1))
        ndigits = max(ndigits, 5)
        fmt = "%0$(ndigits)d"
        paths = String[]
        for i in 0:n-1
            shard_suffix = lpad(string(i), ndigits, '0') * "-of-" * lpad(string(n), ndigits, '0')
            push!(paths, "$(prefix)-$(shard_suffix)")
        end
        return paths
    end

    # Pattern: prefix-NNNNN-of-MMMMM (already a shard path)
    m = match(r"^(.+)-(\d{5})-of-(\d{5})$", path)
    if m !== nothing
        prefix = String(m.captures[1])
        total  = parse(Int, m.captures[3])
        ndigits = 5
        paths = String[]
        for i in 0:total-1
            shard_suffix = lpad(string(i), ndigits, '0') * "-of-" * lpad(string(total), ndigits, '0')
            push!(paths, "$(prefix)-$(shard_suffix)")
        end
        return paths
    end

    return nothing
end

"""
    is_sharded_path(path::String) -> Bool

Return true if path matches a shard pattern.
"""
function is_sharded_path(path::String)::Bool
    return get_sharded_paths(path) !== nothing
end

"""
    merge_jackhmmer_results(results::Vector{String}) -> String

Merge multiple A3M outputs (one per shard), deduplicating by sequence content.
The query sequence (from the first result) is always preserved at position 1.
"""
function merge_jackhmmer_results(results::Vector{String})::String
    isempty(results) && return ""
    msas = Msa[Msa_from_a3m(r) for r in results if !isempty(strip(r))]
    isempty(msas) && return ""
    merged = merge_msas(msas)
    return msa_to_a3m(merged)
end

"""
    merge_nhmmer_results(results::Vector{String}) -> String

Merge multiple Stockholm outputs (one per shard), deduplicating by sequence content.
"""
function merge_nhmmer_results(results::Vector{String})::String
    isempty(results) && return ""
    msas = Msa[Msa_from_stockholm(r) for r in results if !isempty(strip(r))]
    isempty(msas) && return ""
    merged = merge_msas(msas)
    # Return as Stockholm format
    buf = IOBuffer()
    println(buf, "# STOCKHOLM 1.0")
    for (desc, seq) in zip(merged.descriptions, merged.sequences)
        println(buf, "$(desc)\t$(seq)")
    end
    println(buf, "//")
    return String(take!(buf))
end

"""
    run_parallel_shards(
        shard_paths::Vector{String},
        run_fn::Function,
        max_parallel::Union{Int,Nothing}
    ) -> Vector{String}

Run run_fn(shard_path) for each shard in parallel (up to max_parallel at once).
Returns a vector of output strings in the same order as shard_paths.
"""
function run_parallel_shards(
    shard_paths::Vector{String},
    run_fn::Function,
    max_parallel::Union{Int,Nothing}
)::Vector{String}
    n = length(shard_paths)
    n == 0 && return String[]

    actual_parallel = max_parallel === nothing ? n : min(max_parallel, n)
    results = Vector{String}(undef, n)

    if actual_parallel == 1 || n == 1
        for (i, path) in enumerate(shard_paths)
            results[i] = run_fn(path)
        end
        return results
    end

    # Use Julia's @sync/@async for parallel execution
    # We chunk into batches of size actual_parallel
    for chunk_start in 1:actual_parallel:n
        chunk_end = min(chunk_start + actual_parallel - 1, n)
        tasks = Vector{Task}(undef, chunk_end - chunk_start + 1)
        for (j, i) in enumerate(chunk_start:chunk_end)
            local ii = i
            local path = shard_paths[ii]
            tasks[j] = @async begin
                try
                    run_fn(path)
                catch e
                    @error "Error processing shard $path" exception=e
                    ""
                end
            end
        end
        for (j, i) in enumerate(chunk_start:chunk_end)
            results[i] = fetch(tasks[j])
        end
    end

    return results
end
