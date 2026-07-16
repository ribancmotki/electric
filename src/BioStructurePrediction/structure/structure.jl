"""
Core structure types and operations.
"""

using Printf
using LinearAlgebra

# ──────────────────────────────────────────────
#  Supporting types
# ──────────────────────────────────────────────

"""
    ChainInfo

Metadata for a single chain within a Structure.
"""
struct ChainInfo
    chain_id::String
    first_atom_idx::Int   # 1-based index into atom table
    last_atom_idx::Int
end

"""
    ResidueInfo

Metadata for a single residue within a Structure.
"""
struct ResidueInfo
    chain_id::String
    seq_id::String
    comp_id::String
    first_atom_idx::Int
end

"""
    Bond

A covalent bond between two atoms in a structure.
"""
struct Bond
    atom1_idx::Int
    atom2_idx::Int
    bond_order::Int
end

# ──────────────────────────────────────────────
#  Structure
# ──────────────────────────────────────────────

"""
    Structure

Top-level container for a biomolecular structure.
"""
struct Structure
    name::String
    atoms::StructureTable
    chains::Vector{ChainInfo}
    residues::Vector{ResidueInfo}
    bonds::Vector{Bond}
end

# ──────────────────────────────────────────────
#  Basic accessors
# ──────────────────────────────────────────────

"""
    num_atoms(s::Structure) -> Int
"""
function num_atoms(s::Structure)::Int
    return nrows(s.atoms)
end

"""
    num_residues(s::Structure) -> Int
"""
function num_residues(s::Structure)::Int
    return length(s.residues)
end

"""
    num_chains(s::Structure) -> Int
"""
function num_chains(s::Structure)::Int
    return length(s.chains)
end

"""
    atom_positions(s::Structure) -> Matrix{Float32}

Return all atom positions as a (num_atoms, 3) Float32 matrix.
"""
function atom_positions(s::Structure)::Matrix{Float32}
    n = num_atoms(s)
    pos = Matrix{Float32}(undef, n, 3)
    pos[:, 1] = get_column(s.atoms, :Cartn_x)
    pos[:, 2] = get_column(s.atoms, :Cartn_y)
    pos[:, 3] = get_column(s.atoms, :Cartn_z)
    return pos
end

"""
    atom_mask(s::Structure) -> Vector{Bool}

Return a mask indicating which atoms are present (all true for a complete structure).
"""
function atom_mask(s::Structure)::Vector{Bool}
    n = num_atoms(s)
    mask = trues(n)
    xs = get_column(s.atoms, :Cartn_x)
    ys = get_column(s.atoms, :Cartn_y)
    zs = get_column(s.atoms, :Cartn_z)
    for i in 1:n
        if xs[i] == 0f0 && ys[i] == 0f0 && zs[i] == 0f0
            mask[i] = false
        end
    end
    return mask
end

"""
    get_chain(s::Structure, chain_id::String) -> Structure

Return a sub-Structure containing only atoms from the specified chain.
"""
function get_chain(s::Structure, chain_id::String)::Structure
    chain_ids = get_column(s.atoms, :label_asym_id)
    mask = BitVector(chain_ids .== chain_id)
    sub_atoms = filter_rows(s.atoms, mask)
    sub_chains = filter(c -> c.chain_id == chain_id, s.chains)
    sub_residues = filter(r -> r.chain_id == chain_id, s.residues)
    # Re-index bonds
    atom_indices = findall(mask)
    idx_map = Dict(old => new for (new, old) in enumerate(atom_indices))
    sub_bonds = Bond[]
    for b in s.bonds
        if haskey(idx_map, b.atom1_idx) && haskey(idx_map, b.atom2_idx)
            push!(sub_bonds, Bond(idx_map[b.atom1_idx], idx_map[b.atom2_idx], b.bond_order))
        end
    end
    return Structure("$(s.name)_$chain_id", sub_atoms, sub_chains, sub_residues, sub_bonds)
end

"""
    get_residue(s::Structure, chain_id::String, res_idx::Int) -> Structure

Return a sub-Structure containing only atoms from the specified residue (1-based index).
"""
function get_residue(s::Structure, chain_id::String, res_idx::Int)::Structure
    chain_residues = filter(r -> r.chain_id == chain_id, s.residues)
    res_idx > length(chain_residues) && error("Residue index $res_idx out of range for chain $chain_id")
    res = chain_residues[res_idx]
    chain_ids = get_column(s.atoms, :label_asym_id)
    seq_ids   = get_column(s.atoms, :label_seq_id)
    mask = BitVector((chain_ids .== chain_id) .& (seq_ids .== res.seq_id))
    sub_atoms = filter_rows(s.atoms, mask)
    return Structure("$(s.name)_$(chain_id)_$(res.seq_id)", sub_atoms, ChainInfo[], [res], Bond[])
end

# ──────────────────────────────────────────────
#  Construction from coordinates
# ──────────────────────────────────────────────

"""
    structure_from_arrays(;
        name, chain_ids, res_ids, comp_ids, atom_names, elements,
        positions, bfactors, seq_ids
    ) -> Structure

Construct a Structure from flat arrays of per-atom data.
"""
function structure_from_arrays(;
    name::String,
    chain_ids::Vector{String},
    res_ids::Vector{String},
    comp_ids::Vector{String},
    atom_names::Vector{String},
    elements::Vector{String},
    positions::Matrix{Float32},   # (n, 3)
    bfactors::Vector{Float32},
    seq_ids::Union{Vector{String},Nothing} = nothing,
    entity_ids::Union{Vector{String},Nothing} = nothing,
)::Structure
    n = length(chain_ids)
    n == size(positions, 1) || error("positions rows ($( size(positions,1) )) != n ($n)")

    res_ids_use    = seq_ids    !== nothing ? seq_ids    : res_ids
    entity_ids_use = entity_ids !== nothing ? entity_ids : fill("1", n)

    groups = String[is_hetatm(comp_ids[i]) ? "HETATM" : "ATOM  " for i in 1:n]

    t = StructureTable()
    add_column!(t, :group_PDB,       groups)
    add_column!(t, :id,              collect(1:n))
    add_column!(t, :type_symbol,     elements)
    add_column!(t, :label_atom_id,   atom_names)
    add_column!(t, :label_comp_id,   comp_ids)
    add_column!(t, :label_asym_id,   chain_ids)
    add_column!(t, :label_entity_id, entity_ids_use)
    add_column!(t, :label_seq_id,    res_ids_use)
    add_column!(t, :Cartn_x,         positions[:, 1])
    add_column!(t, :Cartn_y,         positions[:, 2])
    add_column!(t, :Cartn_z,         positions[:, 3])
    add_column!(t, :occupancy,       fill(1f0, n))
    add_column!(t, :B_iso_or_equiv,  bfactors)
    add_column!(t, :auth_seq_id,     res_ids_use)
    add_column!(t, :auth_asym_id,    chain_ids)

    # Build chain infos
    unique_cids = unique(chain_ids)
    chain_infos = ChainInfo[]
    for cid in unique_cids
        idxs = findall(==(cid), chain_ids)
        push!(chain_infos, ChainInfo(cid, first(idxs), last(idxs)))
    end

    # Build residue infos
    seen = Set{Tuple{String,String}}()
    residue_infos = ResidueInfo[]
    for i in 1:n
        key = (chain_ids[i], res_ids_use[i])
        if key ∉ seen
            push!(seen, key)
            push!(residue_infos, ResidueInfo(chain_ids[i], res_ids_use[i], comp_ids[i], i))
        end
    end

    return Structure(name, t, chain_infos, residue_infos, Bond[])
end

# ──────────────────────────────────────────────
#  I/O convenience
# ──────────────────────────────────────────────

"""
    to_mmcif(s::Structure; bfactors=nothing) -> String

Convert Structure to mmCIF string. Delegates to structure_to_mmcif.
"""
function to_mmcif(s::Structure; bfactors::Union{Vector{Float32},Nothing}=nothing)::String
    return structure_to_mmcif(s; bfactors=bfactors)
end

"""
    from_mmcif(mmcif_str::String) -> Structure

Parse mmCIF string to Structure. Delegates to parse_structure_from_mmcif_string.
"""
function from_mmcif(mmcif_str::String)::Structure
    return parse_structure_from_mmcif_string(mmcif_str)
end
