#!/usr/bin/env julia
"""
run_prediction_data_test.jl — Data pipeline tests.

Tests the featurisation pipeline, MSA parsing, template search stubs,
and structure I/O without requiring external binaries or databases.

Usage:
  julia run_prediction_data_test.jl
"""

using Pkg
Pkg.activate(dirname(@__FILE__))

using Test
using Logging
using Dates

include(joinpath(dirname(@__FILE__), "src", "BioStructurePrediction", "__init__.jl"))
using .BioStructurePrediction

# ──────────────────────────────────────────────
#  Test data
# ──────────────────────────────────────────────

const TEST_FASTA = """>seq1
ACDEFGHIKLMNPQRSTVWY
>seq2
ACDEFGHIKLMNPQRSTVWY
>seq3
ACDE-GHIKLMNPQRSTVWY
"""

const TEST_STOCKHOLM = """# STOCKHOLM 1.0
seq1    ACDEFGHIKLMNPQRSTVWY
seq2    ACDEFGHIKLMNPQRSTVWY
#=GC RF xxxxxxxxxxxxxxxxxxxxxxxxxx
//
"""

const TEST_A3M = """>seq1
ACDEFGHIKLMNPQRSTVWY
>seq2
ACDEfGHIKLMNPQRSTVWY
>seq3
ACDEFGHIKLMNPQRSTVwY
"""

const TEST_JSON_INPUT = """
{
  "name": "test_case",
  "modelSeeds": [42],
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
        "modifications": []
      }
    },
    {
      "ligand": {
        "id": "B",
        "ccdCodes": ["ATP"]
      }
    }
  ],
  "dialect": "alphafold3",
  "version": 1
}
"""

const TEST_MMCIF = """data_TEST
#
_entry.id TEST
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM 1 N N ALA A 1 1.000 2.000 3.000 1.00 10.00
ATOM 2 C CA ALA A 1 2.000 3.000 4.000 1.00 10.00
ATOM 3 C C ALA A 1 3.000 4.000 5.000 1.00 10.00
ATOM 4 O O ALA A 1 4.000 5.000 6.000 1.00 10.00
ATOM 5 N N GLY A 2 5.000 6.000 7.000 1.00 10.00
ATOM 6 C CA GLY A 2 6.000 7.000 8.000 1.00 10.00
#
"""

# ──────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────

@testset "BioStructurePrediction data pipeline tests" begin

    # ── 1. FASTA / A3M / Stockholm parsers ─────────────────────────────────────
    @testset "Sequence parsers" begin
        seqs = parse_fasta(TEST_FASTA)
        @test length(seqs) == 3
        @test seqs[1][1] == "seq1"
        @test seqs[1][2] == "ACDEFGHIKLMNPQRSTVWY"

        seqs_a3m = parse_a3m(TEST_A3M)
        @test length(seqs_a3m) == 3
        # Lowercase insertions should be preserved in A3M
        @test seqs_a3m[2][2][5] == 'f'

        sto_data = parse_stockholm(TEST_STOCKHOLM)
        @test haskey(sto_data, "seq1")
        @test sto_data["seq1"] == "ACDEFGHIKLMNPQRSTVWY"
    end

    # ── 2. MSA construction and truncation ──────────────────────────────────────
    @testset "MSA construction and truncation" begin
        msa = Msa_from_fasta(TEST_FASTA)
        @test n_seqs(msa) == 3
        @test alignment_length(msa) == 20

        msa_trunc = truncate_msa(msa, 2)
        @test n_seqs(msa_trunc) == 2

        # Round-trip through A3M
        a3m_str = msa_to_a3m(msa)
        msa2    = Msa_from_a3m(a3m_str)
        @test n_seqs(msa2) == n_seqs(msa)
    end

    # ── 3. MSA features ──────────────────────────────────────────────────────────
    @testset "MSA features" begin
        msa   = Msa_from_fasta(TEST_FASTA)
        feats = make_msa_features(msa)

        @test haskey(feats, "msa_feat")
        @test haskey(feats, "msa_mask")
        @test size(feats["msa_feat"], 2) == alignment_length(msa)
        @test size(feats["msa_feat"], 3) == MSA_FEAT_DIM

        extra = make_extra_msa_features(msa)
        @test haskey(extra, "extra_msa_feat")
        @test size(extra["extra_msa_feat"], 3) == EXTRA_MSA_FEAT_DIM
    end

    # ── 4. FoldingInput JSON parsing ──────────────────────────────────────────────
    @testset "FoldingInput JSON parsing" begin
        fi = FoldingInput_from_json(TEST_JSON_INPUT)
        @test fi.name == "test_case"
        @test fi.rng_seeds == [42]
        @test length(fi.protein_chains) == 1
        @test fi.protein_chains[1].sequence == "ACDEFGHIKLMNPQRSTVWY"
        @test length(fi.ligands) == 1
        @test fi.ligands[1].ccd_codes == ["ATP"]
        @test fi.dialect == "alphafold3"

        # Round-trip
        json_str = FoldingInput_to_json(fi)
        fi2 = FoldingInput_from_json(json_str)
        @test fi2.name == fi.name
        @test fi2.rng_seeds == fi.rng_seeds
        @test fi2.protein_chains[1].sequence == fi.protein_chains[1].sequence
    end

    # ── 5. FoldingInput seed utility ──────────────────────────────────────────────
    @testset "FoldingInput seed utility" begin
        fi = FoldingInput_from_json(TEST_JSON_INPUT)
        fi_multi = with_multiple_seeds(fi, [1, 2, 3])
        @test fi_multi.rng_seeds == [1, 2, 3]
        @test sanitised_name(fi) == "test_case"
    end

    # ── 6. mmCIF parser ──────────────────────────────────────────────────────────
    @testset "mmCIF parser" begin
        s = parse_structure_from_mmcif_string(TEST_MMCIF)
        @test num_atoms(s) == 6

        chain_A = get_chain(s, "A")
        @test chain_A !== nothing

        # Round-trip
        mmcif_out = to_mmcif(s)
        s2 = parse_structure_from_mmcif_string(mmcif_out)
        @test num_atoms(s2) == num_atoms(s)
    end

    # ── 7. Atom layout utilities ──────────────────────────────────────────────────
    @testset "Atom layout utilities" begin
        n_tokens = 5
        positions = randn(Float32, n_tokens, NUM_ATOM_SLOTS, 3)
        mask      = falses(n_tokens, NUM_ATOM_SLOTS)
        mask[:, 1:3] .= true   # 3 atoms per token = 15 atoms total

        flat_pos, flat_mask = atom_layout_to_flat(positions, mask)
        @test length(flat_mask) == n_tokens * 3
        @test all(flat_mask)
        @test size(flat_pos, 2) == 3

        token_atom_counts = fill(3, n_tokens)
        reconstructed = flat_to_atom_layout(flat_pos, token_atom_counts)
        @test size(reconstructed) == (n_tokens, 3, 3)
        @test reconstructed[1, 1, :] ≈ flat_pos[1, :]
    end

    # ── 8. Relative position encoding ────────────────────────────────────────────
    @testset "Relative position encoding" begin
        token_index    = Int32[1, 2, 3, 4, 5]
        token_chain_ids = ["A", "A", "A", "B", "B"]

        enc = compute_relative_position_encoding(token_index, token_chain_ids)
        @test size(enc, 1) == 5
        @test size(enc, 2) == 5
        @test size(enc, 3) == NUM_RELATIVE_POS_BINS

        # Diagonal (same position) should have rel_pos = 0 → bin = MAX_RELATIVE_IDX + 1
        mid_bin = MAX_RELATIVE_IDX + 1
        @test enc[1, 1, mid_bin] ≈ 1f0
        @test enc[2, 2, mid_bin] ≈ 1f0

        # Same-chain indicator: A-A pairs
        last_bin = NUM_RELATIVE_POS_BINS
        @test enc[1, 2, last_bin] ≈ 1f0   # same chain
        @test enc[1, 4, last_bin] ≈ 0f0   # different chain
    end

    # ── 9. Geometry module ────────────────────────────────────────────────────────
    @testset "Backbone frame construction" begin
        n_res = 3
        # Simple test: straight chain along x-axis
        n_pos  = Float32[0 0 0; 1 0 0; 2 0 0]
        ca_pos = Float32[0.5 0 0; 1.5 0 0; 2.5 0 0]
        c_pos  = Float32[1 0 0; 2 0 0; 3 0 0]

        frames = make_backbone_frames(n_pos, ca_pos, c_pos)
        @test length(frames) == n_res
        @test frames[1] isa Rigid3Array

        # Translation should be CA position
        @test frames[1].translation ≈ ca_pos[1, :] atol=1e-5f0

        # Rotation should be orthonormal
        for f in frames
            @test f.rotation * f.rotation' ≈ Matrix{Float32}(I, 3, 3) atol=1e-5f0
        end
    end

    # ── 10. Chemical component sets ───────────────────────────────────────────────
    @testset "Chemical component sets" begin
        @test classify_component("HOH") == :water
        @test classify_component("NAD") ∈ (:nucleotide, :common_ligand, :unknown)
        @test classify_component("ALA") ∈ (:unknown, :standard_amino_acid)
    end

    # ── 11. TM-score and GDT ─────────────────────────────────────────────────────
    @testset "TM-score and GDT_TS" begin
        n = 20
        true_pos = randn(Float32, n, 3)
        mask     = trues(n)

        tm = compute_tm_score(true_pos, true_pos, mask)
        @test tm ≈ 1.0f0 atol=0.01f0

        gdt = compute_gdt_ts(true_pos, true_pos, mask)
        @test gdt ≈ 1.0f0 atol=0.01f0

        # Shifted structure: should score lower
        shifted_pos = true_pos .+ 5f0
        _, rmsd = superimpose(shifted_pos, true_pos, mask)
        @test rmsd < 0.01f0   # pure translation → perfect superimposition
    end

    # ── 12. Bond adjacency matrix ─────────────────────────────────────────────────
    @testset "Bond adjacency matrix" begin
        s = parse_structure_from_mmcif_string(TEST_MMCIF)
        ccd = Ccd(Dict())   # empty CCD
        adj = bond_adjacency_matrix(s, ccd)
        @test size(adj, 1) == num_atoms(s)
        @test size(adj, 2) == num_atoms(s)
        # Adjacency matrix should be symmetric
        @test adj ≈ adj'
    end

    # ── 13. pLDDT/PAE binning ────────────────────────────────────────────────────
    @testset "pLDDT / PAE bin arithmetic" begin
        n = 4
        logits_plddt = zeros(Float32, n, NUM_ATOM_TYPES_PLDDT, PLDDT_NUM_BINS)
        plddt = compute_plddt(logits_plddt)
        # All-zero logits → uniform distribution → mean ≈ 50
        expected_plddt = mean(PLDDT_BIN_CENTERS) * 100f0
        @test all(abs.(plddt .- expected_plddt) .< 1f0)

        logits_pae = zeros(Float32, n, n, PAE_NUM_BINS)
        pae = compute_pae(logits_pae)
        expected_pae = mean(PAE_BIN_CENTERS)
        @test all(abs.(pae .- expected_pae) .< 0.1f0)
    end

    # ── 14. Periodic table ────────────────────────────────────────────────────────
    @testset "Periodic table" begin
        @test get_atomic_number("C")  == 6
        @test get_atomic_number("N")  == 7
        @test get_atomic_number("O")  == 8
        @test get_atomic_number("FE") == 26
        @test is_metal_element("FE")  == true
        @test is_metal_element("C")   == false
    end

    # ── 15. Sharded path utilities ────────────────────────────────────────────────
    @testset "Sharded path utilities" begin
        @test is_sharded_path("database@10")     == true
        @test is_sharded_path("database")        == false
        paths = get_sharded_paths("database@3")
        @test length(paths) == 3
    end

end  # @testset

@info "All data pipeline tests completed."
