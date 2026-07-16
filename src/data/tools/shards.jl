"""
shards.jl — Sharded database path utilities.
"""

"""
    is_sharded_path(path::String) -> Bool

Return true if the path represents a sharded database (e.g., "db.fasta@4").
"""
function is_sharded_path(path::String)::Bool
    return !isnothing(match(r"@\d+$", path))
end

"""
    get_sharded_paths(path::String) -> Vector{String}

Expand a sharded path "db.fasta@4" to ["db.fasta-0-of-4", "db.fasta-1-of-4", ...].
"""
function get_sharded_paths(path::String)::Vector{String}
    m = match(r"^(.+)@(\d+)$", path)
    m === nothing && return [path]
    base, n_str = m.captures
    n = parse(Int, n_str)
    return ["$base-$i-of-$n" for i in 0:n-1]
end

"""
    merge_jackhmmer_results(results::Vector{String}) -> String

Merge multiple jackhmmer A3M outputs (from separate shards) into one A3M.
Keep query sequence from first shard, deduplicate other sequences.
"""
function merge_jackhmmer_results(results::Vector{String})::String
    isempty(results) && return ""
    length(results) == 1 && return results[1]

    # Parse all A3M records
    all_records = Tuple{String,String}[]
    seen_descs = Set{String}()
    for a3m in results
        for (desc, seq) in a3m_to_fasta(a3m)
            desc in seen_descs && continue
            push!(seen_descs, desc)
            push!(all_records, (desc, seq))
        end
    end

    buf = IOBuffer()
    for (desc, seq) in all_records
        println(buf, ">$desc")
        println(buf, seq)
    end
    return String(take!(buf))
end

"""
    merge_nhmmer_results(results::Vector{String}) -> String

Merge multiple nhmmer A3M outputs.
"""
function merge_nhmmer_results(results::Vector{String})::String
    return merge_jackhmmer_results(results)
end

"""
    run_parallel_shards(f::Function, paths::Vector{String};
                        max_parallel::Union{Int,Nothing}=nothing) -> Vector{Any}

Run function `f` on each shard path in parallel, respecting max_parallel limit.
Returns results in original shard order.
"""
function run_parallel_shards(f::Function, paths::Vector{String};
                              max_parallel::Union{Int,Nothing}=nothing)
    n = length(paths)
    n == 0 && return Any[]
    n == 1 && return [f(paths[1])]

    max_p = max_parallel === nothing ? min(n, Threads.nthreads()) : min(max_parallel, n)

    results = Vector{Any}(undef, n)
    tasks = Task[]

    for (i, path) in enumerate(paths)
        t = Threads.@spawn f(path)
        push!(tasks, t)
        # Throttle: wait if we have too many in flight
        if length(tasks) >= max_p
            for (j, task) in enumerate(tasks)
                results[j] = fetch(task)
            end
            empty!(tasks)
        end
    end
    for (j, task) in enumerate(tasks)
        results[length(results)-length(tasks)+j] = fetch(task)
    end
    return results
end
