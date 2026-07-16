"""
Biological assembly expansion from mmCIF symmetry operations.
"""

using LinearAlgebra

"""
    AssemblyOperation

A single symmetry operation (rotation + translation) for assembly expansion.
"""
struct AssemblyOperation
    id::String
    rotation::Matrix{Float64}     # 3×3
    translation::Vector{Float64}  # length 3
end

"""
    AssemblyGenerator

Specifies which chains and operations make up one biological assembly.
"""
struct AssemblyGenerator
    assembly_id::String
    oper_expression::String
    asym_ids::Vector{String}
end

"""
    parse_assembly_info(mmcif_dict::Dict{String,Any}) -> Tuple{Vector{AssemblyGenerator}, Dict{String,AssemblyOperation}}

Extract assembly generators and symmetry operations from a parsed mmCIF dict.
"""
function parse_assembly_info(mmcif_dict::Dict{String,Any})::Tuple{Vector{AssemblyGenerator}, Dict{String,AssemblyOperation}}
    block = first(values(mmcif_dict))

    generators = AssemblyGenerator[]
    asm_ids   = _get_loop_col(block, "_pdbx_struct_assembly_gen.assembly_id",  String[])
    oper_exprs = _get_loop_col(block, "_pdbx_struct_assembly_gen.oper_expression", String[])
    asym_lists = _get_loop_col(block, "_pdbx_struct_assembly_gen.asym_id_list", String[])

    for (asm_id, oper_expr, asym_list) in zip(asm_ids, oper_exprs, asym_lists)
        asym_ids = String[strip(s) for s in split(asym_list, ",")]
        push!(generators, AssemblyGenerator(asm_id, oper_expr, asym_ids))
    end

    operations = Dict{String,AssemblyOperation}()
    oper_ids   = _get_loop_col(block, "_pdbx_struct_oper_list.id",   String[])
    m11 = _get_loop_col(block, "_pdbx_struct_oper_list.matrix[1][1]", String[])
    m12 = _get_loop_col(block, "_pdbx_struct_oper_list.matrix[1][2]", String[])
    m13 = _get_loop_col(block, "_pdbx_struct_oper_list.matrix[1][3]", String[])
    m21 = _get_loop_col(block, "_pdbx_struct_oper_list.matrix[2][1]", String[])
    m22 = _get_loop_col(block, "_pdbx_struct_oper_list.matrix[2][2]", String[])
    m23 = _get_loop_col(block, "_pdbx_struct_oper_list.matrix[2][3]", String[])
    m31 = _get_loop_col(block, "_pdbx_struct_oper_list.matrix[3][1]", String[])
    m32 = _get_loop_col(block, "_pdbx_struct_oper_list.matrix[3][2]", String[])
    m33 = _get_loop_col(block, "_pdbx_struct_oper_list.matrix[3][3]", String[])
    v1  = _get_loop_col(block, "_pdbx_struct_oper_list.vector[1]", String[])
    v2  = _get_loop_col(block, "_pdbx_struct_oper_list.vector[2]", String[])
    v3  = _get_loop_col(block, "_pdbx_struct_oper_list.vector[3]", String[])

    pf(s) = tryparse(Float64, s) !== nothing ? parse(Float64, s) : 0.0

    for i in eachindex(oper_ids)
        rot = [pf(m11[i]) pf(m12[i]) pf(m13[i]);
               pf(m21[i]) pf(m22[i]) pf(m23[i]);
               pf(m31[i]) pf(m32[i]) pf(m33[i])]
        trans = [pf(v1[i]), pf(v2[i]), pf(v3[i])]
        operations[oper_ids[i]] = AssemblyOperation(oper_ids[i], rot, trans)
    end

    return generators, operations
end

"""
    expand_assembly(s::Structure, asm_gen::AssemblyGenerator, ops::Dict{String,AssemblyOperation}) -> Structure

Apply the symmetry operations in asm_gen to the relevant chains of s and return the expanded assembly.
"""
function expand_assembly(
    s::Structure,
    asm_gen::AssemblyGenerator,
    ops::Dict{String,AssemblyOperation}
)::Structure
    # Parse oper_expression (may be "1" or "(1-3)" or "1,2,3")
    oper_ids = parse_oper_expression(asm_gen.oper_expression)

    all_tables = StructureTable[]
    all_chains = ChainInfo[]
    all_residues = ResidueInfo[]

    chain_ids_col = get_column(s.atoms, :label_asym_id)

    chain_counter = 0
    for oper_id in oper_ids
        op = get(ops, oper_id, nothing)
        op === nothing && (@warn "Operation $oper_id not found"; continue)

        for asym_id in asm_gen.asym_ids
            mask = BitVector(chain_ids_col .== asym_id)
            any(mask) || continue

            sub = filter_rows(s.atoms, mask)
            n_sub = nrows(sub)

            # Apply rotation + translation
            xs = copy(sub.columns[:Cartn_x])
            ys = copy(sub.columns[:Cartn_y])
            zs = copy(sub.columns[:Cartn_z])

            R = op.rotation
            t = op.translation
            for j in 1:n_sub
                v = [Float64(xs[j]), Float64(ys[j]), Float64(zs[j])]
                v2 = R * v .+ t
                xs[j] = Float32(v2[1])
                ys[j] = Float32(v2[2])
                zs[j] = Float32(v2[3])
            end

            new_chain_id = "$(asym_id)_$(oper_id)"
            new_cols = Dict{Symbol,Vector}()
            for col in sub.column_order
                if col == :Cartn_x
                    new_cols[col] = xs
                elseif col == :Cartn_y
                    new_cols[col] = ys
                elseif col == :Cartn_z
                    new_cols[col] = zs
                elseif col == :label_asym_id || col == :auth_asym_id
                    new_cols[col] = fill(new_chain_id, n_sub)
                else
                    new_cols[col] = copy(sub.columns[col])
                end
            end
            new_table = StructureTable(new_cols, copy(sub.column_order))
            push!(all_tables, new_table)
        end
    end

    isempty(all_tables) && return s
    combined = concat(all_tables)
    return Structure("$(s.name)_assembly_$(asm_gen.assembly_id)", combined, ChainInfo[], ResidueInfo[], Bond[])
end

"""
    parse_oper_expression(expr::String) -> Vector{String}

Parse a symmetry operation expression like "1", "1,2,3", or "(1-3)" into operation IDs.
"""
function parse_oper_expression(expr::String)::Vector{String}
    expr = strip(expr)
    # Handle (1-3) range notation
    m = match(r"^\((\d+)-(\d+)\)$", expr)
    if m !== nothing
        lo = parse(Int, m.captures[1])
        hi = parse(Int, m.captures[2])
        return string.(lo:hi)
    end
    # Handle comma-separated or single
    return String[strip(s) for s in split(expr, ",")]
end
