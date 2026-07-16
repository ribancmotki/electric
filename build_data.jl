#!/usr/bin/env julia
"""
build_data.jl — Compile the Chemical Component Dictionary (CCD) mmCIF file
into a binary database for fast loading during inference.

Usage:
  julia build_data.jl --ccd_cif <components.cif> [--output_dir <dir>]
"""

using Pkg
Pkg.activate(dirname(@__FILE__))

using ArgParse
using Logging
using CodecZstd
using JSON3
using Serialization

include(joinpath(dirname(@__FILE__), "src", "BioStructurePrediction", "__init__.jl"))
using .BioStructurePrediction

function parse_commandline()
    s = ArgParseSettings(
        prog        = "build_data.jl",
        description = "Compile the CCD mmCIF into a binary database",
    )
    @add_arg_table! s begin
        "--ccd_cif"
            help     = "Path to the CCD mmCIF file (e.g. components.cif or components.cif.gz)"
            arg_type = String
            required = true
        "--output_dir"
            help     = "Output directory for the compiled database"
            arg_type = String
            default  = get(ENV, "XDG_DATA_HOME", joinpath(homedir(), ".local", "share", "BioStructurePrediction"))
        "--compress"
            help     = "Compress the output database with zstd"
            action   = :store_true
        "--verbose"
            help     = "Print detailed progress"
            action   = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()

    if args["verbose"]
        global_logger(ConsoleLogger(stderr, Logging.Debug))
    else
        global_logger(ConsoleLogger(stderr, Logging.Info))
    end

    ccd_cif_path = args["ccd_cif"]
    output_dir   = args["output_dir"]
    mkpath(output_dir)

    @info "Building CCD database from: $ccd_cif_path"
    @info "Output directory: $output_dir"

    # Read CIF (optionally decompressing .gz)
    cif_content = if endswith(ccd_cif_path, ".gz")
        @info "Decompressing $ccd_cif_path ..."
        using GZip
        GZip.open(ccd_cif_path, "r") do io
            read(io, String)
        end
    else
        read(ccd_cif_path, String)
    end

    @info "Parsing CCD mmCIF ($(round(sizeof(cif_content) / 1_000_000; digits=1)) MB) ..."
    t_parse = @elapsed begin
        ccd = parse_ccd_mmcif(cif_content)
    end
    @info "Parsed $(length(ccd.components)) components in $(round(t_parse; digits=1)) s"

    # Write binary serialisation
    db_path = joinpath(output_dir, "ccd.db")

    if args["compress"]
        db_path_zst = db_path * ".zst"
        @info "Serialising and compressing to $db_path_zst ..."
        io_buf = IOBuffer()
        serialize(io_buf, ccd)
        raw_bytes = take!(io_buf)
        open(db_path_zst, "w") do io
            stream = ZstdCompressorStream(io; level=3)
            write(stream, raw_bytes)
            close(stream)
        end
        @info "Database written: $db_path_zst ($(round(stat(db_path_zst).size / 1_000_000; digits=1)) MB)"
    else
        @info "Serialising to $db_path ..."
        open(db_path, "w") do io
            serialize(io, ccd)
        end
        @info "Database written: $db_path ($(round(stat(db_path).size / 1_000_000; digits=1)) MB)"
    end

    # Also write a JSON index of component IDs for quick lookup
    idx_path = joinpath(output_dir, "ccd_index.json")
    component_ids = sort(collect(keys(ccd.components)))
    open(idx_path, "w") do io
        JSON3.write(io, Dict("component_ids" => component_ids, "count" => length(component_ids)))
    end
    @info "Component index written: $idx_path ($(length(component_ids)) entries)"

    @info "Done."
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
