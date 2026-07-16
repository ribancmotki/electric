"""
Testing utilities for structure-related tests.
"""

using Test

"""
    make_dummy_structure(; n_residues=10, chain_id="A") -> Structure

Create a dummy Structure with n_residues ALA residues for testing purposes.
All atoms are placed at the origin.
"""
function make_dummy_structure(; n_residues::Int=10, chain_id::String="A")::Structure
    atom_names_ala = ATOM_ORDER["ALA"]
    valid_atoms    = filter(!isempty, atom_names_ala)  # N, CA, C, O, CB
    n_atoms_per_res = length(valid_atoms)

    n_total  = n_residues * n_atoms_per_res
    chain_ids  = fill(chain_id, n_total)
    comp_ids   = fill("ALA", n_total)
    res_ids    = vcat([fill(string(i), n_atoms_per_res) for i in 1:n_residues]...)
    atom_names = vcat([valid_atoms for _ in 1:n_residues]...)
    elements   = String[startswith(a, "N") ? "N" : startswith(a, "O") ? "O" :
                        startswith(a, "S") ? "S" : "C" for a in atom_names]

    # Simple backbone coordinates
    positions = zeros(Float32, n_total, 3)
    for i in 1:n_residues
        base_x = Float32((i - 1) * 3.8)  # ~3.8 Å per residue
        for j in 1:n_atoms_per_res
            idx = (i-1)*n_atoms_per_res + j
            positions[idx, 1] = base_x + Float32(j) * 0.5f0
            positions[idx, 2] = Float32(j) * 0.3f0
            positions[idx, 3] = 0f0
        end
    end

    return structure_from_arrays(;
        name       = "dummy",
        chain_ids  = chain_ids,
        res_ids    = res_ids,
        comp_ids   = comp_ids,
        atom_names = atom_names,
        elements   = elements,
        positions  = positions,
        bfactors   = zeros(Float32, n_total),
    )
end

"""
    assert_structure_valid(s::Structure)

Assert basic validity of a Structure object.
"""
function assert_structure_valid(s::Structure)
    @test num_atoms(s) >= 0
    @test num_residues(s) >= 0
    @test num_chains(s) >= 0
    if num_atoms(s) > 0
        pos = atom_positions(s)
        @test size(pos) == (num_atoms(s), 3)
        @test eltype(pos) == Float32
    end
end

"""
    assert_mmcif_roundtrip(s::Structure)

Assert that a Structure can be serialised to mmCIF and parsed back without data loss.
"""
function assert_mmcif_roundtrip(s::Structure)
    mmcif_str = to_mmcif(s)
    s2 = from_mmcif(mmcif_str)
    @test num_atoms(s2) == num_atoms(s)
    @test num_chains(s2) == num_chains(s)
    pos1 = atom_positions(s)
    pos2 = atom_positions(s2)
    @test maximum(abs.(pos1 .- pos2)) < 0.001f0
end
