"""
mapping.jl — Sharded/batched application utilities.
"""

"""
    sharded_apply(fun, shard_size, in_axes, out_axes, args...)

Apply `fun` to `args` in shards of `shard_size` along `in_axes`,
concatenating outputs along `out_axes`.
"""
function sharded_apply(fun, shard_size::Int, in_axes::Tuple, out_axes::Tuple, args...)
    # Determine total size along the shard axis
    first_arr = args[findfirst(a -> a isa AbstractArray, args)]
    ax = in_axes[1]
    ax_norm = ax < 0 ? ndims(first_arr) + ax + 1 : ax
    total = size(first_arr, ax_norm)
    shard_size <= 0 && (shard_size = total)

    results = []
    for start in 1:shard_size:total
        stop = min(start + shard_size - 1, total)
        shard_args = map(args) do arg
            if arg isa AbstractArray
                idxs = ntuple(i -> i == ax_norm ? (start:stop) : Colon(), ndims(arg))
                arg[idxs...]
            else
                arg
            end
        end
        push!(results, fun(shard_args...))
    end

    isempty(results) && return fun(args...)
    out_ax = out_axes[1]
    out_ax_norm = out_ax < 0 ? ndims(results[1]) + out_ax + 1 : out_ax
    return cat(results...; dims=out_ax_norm)
end

"""
    sharded_map(fun, shard_size::Int, in_axes::Tuple, out_axes::Tuple)

Return a function that applies `fun` in shards.
"""
function sharded_map(fun, shard_size::Int, in_axes::Tuple, out_axes::Tuple)
    return args -> sharded_apply(fun, shard_size, in_axes, out_axes, args...)
end

"""
    inference_subbatch(module_fn, subbatch_size::Int,
                       batched_args, nonbatched_args;
                       input_subbatch_dim::Int=1,
                       output_subbatch_dim=nothing)

Run `module_fn` in subbatches to reduce memory usage.
"""
function inference_subbatch(module_fn, subbatch_size::Int,
                             batched_args, nonbatched_args;
                             input_subbatch_dim::Int=1,
                             output_subbatch_dim=nothing)
    isempty(batched_args) && return module_fn(batched_args, nonbatched_args)

    first_arg = batched_args[1]
    total = size(first_arg, input_subbatch_dim)
    subbatch_size <= 0 && (subbatch_size = total)

    out_dim = output_subbatch_dim !== nothing ? output_subbatch_dim : input_subbatch_dim

    results = []
    for start in 1:subbatch_size:total
        stop = min(start + subbatch_size - 1, total)
        sub_batched = map(batched_args) do arg
            if arg isa AbstractArray && size(arg, input_subbatch_dim) == total
                idxs = ntuple(i -> i == input_subbatch_dim ? (start:stop) : Colon(),
                              ndims(arg))
                arg[idxs...]
            else
                arg
            end
        end
        push!(results, module_fn(sub_batched, nonbatched_args))
    end

    isempty(results) && return module_fn(batched_args, nonbatched_args)
    return cat(results...; dims=out_dim)
end

"""
    get_shard_size(n_residues::Int,
                   shard_spec::Vector{Tuple{Union{Int,Nothing},Union{Int,Nothing}}}) -> Int

Look up the appropriate shard size given the number of residues.
Shard spec: [(max_residues_threshold, shard_size), ...]; last entry used if threshold is nothing.
"""
function get_shard_size(n_residues::Int,
                         shard_spec::Vector{<:Tuple})::Int
    for (threshold, shard_sz) in shard_spec
        if threshold === nothing || n_residues <= threshold
            return shard_sz !== nothing ? shard_sz : typemax(Int)
        end
    end
    return typemax(Int)
end
