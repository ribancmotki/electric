"""
bioassemblies.jl — Biological assembly parsing and expansion.
"""

using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

"""
    AssemblyOperation

A single symmetry operation: a 3×3 rotation matrix and 3-vector translation.
"""
struct AssemblyOperation
    id::String
    rotation::Matrix{Float32}     # 3×3
    translation::Vector{Float32}  # 3
end

function Base.show(io::IO, op::AssemblyOperation)
    print(io, "AssemblyOperation(id=$(op.id))")
end

"""
    AssemblyGenerator

Specifies which chains and which operations to use for a biological assembly.
"""
struct AssemblyGenerator
    assembly_id::String
    chain_ids::Vector{String}
    operation_ids::Vector{String}  # IDs of AssemblyOperations to apply
end

# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────

"""
    parse_assembly_info(block::CifDict) -> Tuple{Vector{AssemblyGenerator}, Dict{String,AssemblyOperation}}

Parse biological assembly information from an mmCIF block.
Returns (generators, operations).
"""
function parse_assembly_info(block::CifDict)
    # Parse operations
    oper_ids = get_loop_col(block, "_pdbx_struct_oper_list.id")
    m11 = get_loop_col(block, "_pdbx_struct_oper_list.matrix[1][1]")
    m12 = get_loop_col(block, "_pdbx_struct_oper_list.matrix[1][2]")
    m13 = get_loop_col(block, "_pdbx_struct_oper_list.matrix[1][3]")
    m21 = get_loop_col(block, "_pdbx_struct_oper_list.matrix[2][1]")
    m22 = get_loop_col(block, "_pdbx_struct_oper_list.matrix[2][2]")
    m23 = get_loop_col(block, "_pdbx_struct_oper_list.matrix[2][3]")
    m31 = get_loop_col(block, "_pdbx_struct_oper_list.matrix[3][1]")
    m32 = get_loop_col(block, "_pdbx_struct_oper_list.matrix[3][2]")
    m33 = get_loop_col(block, "_pdbx_struct_oper_list.matrix[3][3]")
    v1  = get_loop_col(block, "_pdbx_struct_oper_list.vector[1]")
    v2  = get_loop_col(block, "_pdbx_struct_oper_list.vector[2]")
    v3  = get_loop_col(block, "_pdbx_struct_oper_list.vector[3]")

    ops = Dict{String,AssemblyOperation}()
    for i in 1:length(oper_ids)
        rot = Float32[
            _pf32(m11, i) _pf32(m12, i) _pf32(m13, i);
            _pf32(m21, i) _pf32(m22, i) _pf32(m23, i);
            _pf32(m31, i) _pf32(m32, i) _pf32(m33, i);
        ]
        trans = Float32[_pf32(v1, i), _pf32(v2, i), _pf32(v3, i)]
        ops[oper_ids[i]] = AssemblyOperation(oper_ids[i], rot, trans)
    end

    # Parse assembly generators
    asm_ids   = get_loop_col(block, "_pdbx_struct_assembly_gen.assembly_id")
    oper_exprs= get_loop_col(block, "_pdbx_struct_assembly_gen.oper_expression")
    asym_lists= get_loop_col(block, "_pdbx_struct_assembly_gen.asym_id_list")

    generators = AssemblyGenerator[]
    for i in 1:length(asm_ids)
        asym_str = get(asym_lists, i, "")
        chain_ids = String[strip(c) for c in split(asym_str, ',')]
        oper_str  = get(oper_exprs, i, "1")
        oper_ids_list = _parse_oper_expression(oper_str)
        push!(generators, AssemblyGenerator(asm_ids[i], chain_ids, oper_ids_list))
    end

    return generators, ops
end

"""
    expand_assembly(s::Structure, assembly_id::String,
                    generators::Vector{AssemblyGenerator},
                    operations::Dict{String,AssemblyOperation}) -> Structure

Apply biological assembly transformations to a structure.
"""
function expand_assembly(s::Structure, assembly_id::String,
                          generators::Vector{AssemblyGenerator},
                          operations::Dict{String,AssemblyOperation})::Structure
    relevant = filter(g -> g.assembly_id == assembly_id, generators)
    isempty(relevant) && return s

    parts = Structure[]
    copy_num = 0

    for gen in relevant
        for op_id in gen.operation_ids
            op = get(operations, op_id, nothing)
            op === nothing && continue
            copy_num += 1

            for chain in gen.chain_ids
                chain_s = Base.filter(s; chain_id=chain)
                length(chain_s) == 0 && continue

                # Apply transformation
                n = length(chain_s)
                xyzs = hcat(chain_s.atom_x, chain_s.atom_y, chain_s.atom_z)  # n×3
                xyzs_new = (op.rotation * xyzs') .+ op.translation  # 3×n
                new_x = xyzs_new[1, :]
                new_y = xyzs_new[2, :]
                new_z = xyzs_new[3, :]

                # Rename chain id to avoid conflicts
                new_chain_id = fill(chain * string(copy_num), n)
                new_s = Structure(
                    atom_name    = chain_s.atom_name,
                    atom_element = chain_s.atom_element,
                    res_name     = chain_s.res_name,
                    res_id       = chain_s.res_id,
                    chain_id     = new_chain_id,
                    chain_type   = chain_s.chain_type,
                    atom_x       = Float32.(new_x),
                    atom_y       = Float32.(new_y),
                    atom_z       = Float32.(new_z),
                    atom_b_factor   = chain_s.atom_b_factor,
                    atom_occupancy  = chain_s.atom_occupancy,
                    name         = s.name,
                )
                push!(parts, new_s)
            end
        end
    end

    isempty(parts) && return s
    return _concat_structures(parts, s.name, s.release_date)
end

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

function _pf32(v::Vector{String}, i::Int)::Float32
    i > length(v) && return 0f0
    try; return parse(Float32, v[i]); catch; return 0f0; end
end

function _parse_oper_expression(expr::String)::Vector{String}
    # Handles "1", "1,2,3", "(1-3)", "(1,2)(3,4)", "P 21 21 21", etc.
    # For simplicity, extract all digit sequences
    ms = collect(eachmatch(r"\d+", expr))
    return [m.match for m in ms]
end

function _concat_structures(parts::Vector{Structure}, name::String,
                             release_date::Union{String,Nothing})::Structure
    return Structure(
        atom_name    = vcat([p.atom_name for p in parts]...),
        atom_element = vcat([p.atom_element for p in parts]...),
        res_name     = vcat([p.res_name for p in parts]...),
        res_id       = vcat([p.res_id for p in parts]...),
        chain_id     = vcat([p.chain_id for p in parts]...),
        chain_type   = vcat([p.chain_type for p in parts]...),
        atom_x       = vcat([p.atom_x for p in parts]...),
        atom_y       = vcat([p.atom_y for p in parts]...),
        atom_z       = vcat([p.atom_z for p in parts]...),
        atom_b_factor   = vcat([p.atom_b_factor for p in parts]...),
        atom_occupancy  = vcat([p.atom_occupancy for p in parts]...),
        name         = name,
        release_date = release_date,
    )
end
