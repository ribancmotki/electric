"""
ccd_pickle_gen.jl — Convert the CCD mmCIF file to a binary Julia-serialized format.

Usage:
  julia ccd_pickle_gen.jl --ccd_mmcif /path/to/components.cif --output /path/to/ccd.bin
"""

module CcdPickleGen

using ArgParse
using Serialization
using Logging

# Include parent module constants
include(joinpath(@__DIR__, "..", "chemical_components.jl"))

"""
    build_ccd_from_mmcif(mmcif_path::String) -> Dict{String, CcdComponent}

Parse the full CCD mmCIF file and return a Dict of CcdComponent objects.
The CCD mmCIF file (components.cif from wwPDB) contains one data block per component.
"""
function build_ccd_from_mmcif(mmcif_path::String)::Dict{String,CcdComponent}
    @info "Parsing CCD mmCIF from $mmcif_path"
    components = Dict{String,CcdComponent}()
    text = read(mmcif_path, String)
    blocks = _split_ccd_cif_blocks(text)
    @info "Found $(length(blocks)) CCD blocks"
    count = 0
    for block in blocks
        comp = _parse_ccd_block(block)
        if comp !== nothing
            components[comp.comp_id] = comp
            count += 1
        end
    end
    @info "Successfully parsed $count components"
    return components
end

"""
    write_ccd_binary(components::Dict{String, CcdComponent}, output_path::String)

Serialize the CCD components dict to a binary file using Julia's Serialization.
"""
function write_ccd_binary(components::Dict{String,CcdComponent}, output_path::String)
    @info "Writing CCD binary to $output_path"
    open(output_path, "w") do io
        serialize(io, components)
    end
    @info "Wrote $(length(components)) components to $output_path"
end

function main()
    s = ArgParseSettings(description="Build CCD binary from mmCIF")
    @add_arg_table! s begin
        "--ccd_mmcif"
            help = "Path to CCD mmCIF file (components.cif)"
            required = true
        "--output"
            help = "Output path for binary file"
            default = "ccd.bin"
    end
    args = parse_args(s)
    components = build_ccd_from_mmcif(args["ccd_mmcif"])
    write_ccd_binary(components, args["output"])
end

end # module CcdPickleGen

if abspath(PROGRAM_FILE) == @__FILE__
    CcdPickleGen.main()
end
