"""
Bond detection and manipulation for biomolecular structures.
"""

"""
    get_covalent_bonds(s::Structure, ccd::Ccd) -> Vector{Bond}

Identify covalent bonds in the structure using CCD bond definitions.
Returns bonds between atoms of the same residue, plus polymer backbone bonds.
"""
function get_covalent_bonds(s::Structure, ccd::Ccd)::Vector{Bond}
    bonds = Bond[]
    n = num_atoms(s)
    n == 0 && return bonds

    comp_ids   = get_column(s.atoms, :label_comp_id)
    atom_names = get_column(s.atoms, :label_atom_id)
    chain_ids  = get_column(s.atoms, :label_asym_id)
    seq_ids    = get_column(s.atoms, :label_seq_id)

    # Build index: (chain_id, seq_id, atom_name) → atom_idx
    atom_index = Dict{Tuple{String,String,String},Int}()
    for i in 1:n
        atom_index[(chain_ids[i], seq_ids[i], atom_names[i])] = i
    end

    # Intra-residue bonds from CCD
    residue_groups = Dict{Tuple{String,String,String},Vector{Int}}()
    for i in 1:n
        key = (chain_ids[i], seq_ids[i], comp_ids[i])
        push!(get!(residue_groups, key, Int[]), i)
    end

    for ((cid, sid, cmpid), atom_idxs) in residue_groups
        comp = get_component(ccd, cmpid)
        comp === nothing && continue
        local_name_to_idx = Dict{String,Int}()
        for i in atom_idxs
            local_name_to_idx[atom_names[i]] = i
        end
        for bond in comp.bonds
            a1 = get(local_name_to_idx, bond.atom_id_1, nothing)
            a2 = get(local_name_to_idx, bond.atom_id_2, nothing)
            (a1 !== nothing && a2 !== nothing) && push!(bonds, Bond(a1, a2, bond.bond_order))
        end
    end

    # Inter-residue backbone bonds (protein: C→N; RNA/DNA: O3'→P)
    unique_chains = unique(chain_ids)
    for cid in unique_chains
        chain_residues = filter(r -> r.chain_id == cid, s.residues)
        sort!(chain_residues, by = r -> tryparse(Int, r.seq_id) === nothing ? 0 : parse(Int, r.seq_id))
        for j in 1:length(chain_residues)-1
            r1 = chain_residues[j]
            r2 = chain_residues[j+1]
            # Protein peptide bond: C[r1] → N[r2]
            c_idx = get(atom_index, (cid, r1.seq_id, "C"), nothing)
            n_idx = get(atom_index, (cid, r2.seq_id, "N"), nothing)
            if c_idx !== nothing && n_idx !== nothing
                push!(bonds, Bond(c_idx, n_idx, 1))
            end
            # Nucleotide phosphodiester: O3'[r1] → P[r2]
            o3p_idx = get(atom_index, (cid, r1.seq_id, "O3'"), nothing)
            p_idx   = get(atom_index, (cid, r2.seq_id, "P"),   nothing)
            if o3p_idx !== nothing && p_idx !== nothing
                push!(bonds, Bond(o3p_idx, p_idx, 1))
            end
        end
    end

    # Deduplicate (keep lower-index first)
    seen = Set{Tuple{Int,Int}}()
    unique_bonds = Bond[]
    for b in bonds
        key = b.atom1_idx < b.atom2_idx ? (b.atom1_idx, b.atom2_idx) : (b.atom2_idx, b.atom1_idx)
        if key ∉ seen
            push!(seen, key)
            push!(unique_bonds, Bond(key[1], key[2], b.bond_order))
        end
    end
    return unique_bonds
end

"""
    get_bonded_atom_pairs(s::Structure, bonded_atom_pairs_spec::Vector) -> Vector{Bond}

Resolve explicit bonded atom pair specifications into Bond objects.
Each spec element is a BondedAtomPair.
"""
function get_bonded_atom_pairs(s::Structure, bonded_atom_pairs_spec::Vector)::Vector{Bond}
    bonds = Bond[]
    n = num_atoms(s)
    n == 0 && return bonds

    chain_ids  = get_column(s.atoms, :label_asym_id)
    seq_ids    = get_column(s.atoms, :label_seq_id)
    atom_names = get_column(s.atoms, :label_atom_id)

    atom_index = Dict{Tuple{String,String,String},Int}()
    for i in 1:n
        atom_index[(chain_ids[i], string(seq_ids[i]), atom_names[i])] = i
    end

    for spec in bonded_atom_pairs_spec
        a1 = get(atom_index, (spec.chain1, string(spec.res1), spec.atom1), nothing)
        a2 = get(atom_index, (spec.chain2, string(spec.res2), spec.atom2), nothing)
        if a1 === nothing
            @warn "Bonded atom not found: chain=$(spec.chain1) res=$(spec.res1) atom=$(spec.atom1)"
            continue
        end
        if a2 === nothing
            @warn "Bonded atom not found: chain=$(spec.chain2) res=$(spec.res2) atom=$(spec.atom2)"
            continue
        end
        push!(bonds, Bond(a1, a2, 1))
    end
    return bonds
end

"""
    bond_adjacency_matrix(bonds::Vector{Bond}, n_atoms::Int) -> Matrix{Float32}

Construct a symmetric adjacency matrix from a list of bonds.
Entry [i,j] = 1.0 if atoms i and j are bonded.
"""
function bond_adjacency_matrix(bonds::Vector{Bond}, n_atoms::Int)::Matrix{Float32}
    adj = zeros(Float32, n_atoms, n_atoms)
    for b in bonds
        adj[b.atom1_idx, b.atom2_idx] = 1f0
        adj[b.atom2_idx, b.atom1_idx] = 1f0
    end
    return adj
end
