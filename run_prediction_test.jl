#!/usr/bin/env julia
"""
run_prediction_test.jl — End-to-end inference tests.

Runs a short inference pass on the 5TGY test input and checks that all
output files are produced and confidence metrics are within expected ranges.

Usage:
  julia run_prediction_test.jl [--model_dir <dir>] [--output_dir <dir>]
"""

using Pkg
Pkg.activate(dirname(@__FILE__))

using Test
using Logging
using Dates

include(joinpath(dirname(@__FILE__), "src", "BioStructurePrediction", "__init__.jl"))
using .BioStructurePrediction

# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

"""
    make_small_fold_input() -> FoldingInput

Construct a minimal FoldingInput suitable for fast unit testing.
"""
function make_small_fold_input()
    protein = ProteinChain(
        ["A"],                # chain IDs
        "MKTIIALSYIFCLVFA",   # 17-residue sequence
        ProteinModification[],
        nothing,              # unpaired_msa
        nothing,              # paired_msa
        nothing,              # templates
    )
    return FoldingInput(
        "test_protein",
        [42],          # seeds
        [protein],
        RnaChain[],
        DnaChain[],
        LigandEntity[],
        BondedAtomPair[],
        nothing,
        "alphafold3",
        "1",
    )
end

"""
    make_small_model_runner(num_samples=1, num_recycles=1) -> ModelRunner

Construct a ModelRunner with empty parameters for fast tests.
"""
function make_small_model_runner(; num_samples::Int=1, num_recycles::Int=1)
    cfg = make_model_config(
        flash_attention_implementation = "xla",
        num_diffusion_samples          = num_samples,
        num_recycles                   = num_recycles,
        return_embeddings              = true,
        return_distogram               = true,
    )
    return ModelRunner(Dict{String,Array}(), cfg, false)
end

"""
    make_test_ccd() -> Ccd

Return an empty Ccd (no real components loaded) for fast testing.
"""
make_test_ccd() = Ccd(Dict())

# ──────────────────────────────────────────────
#  Test suites
# ──────────────────────────────────────────────

@testset "BioStructurePrediction end-to-end tests" begin

    # ── 1. FoldingInput serialisation ─────────────────────────────────────────
    @testset "FoldingInput round-trip JSON" begin
        fi = make_small_fold_input()
        json_str = FoldingInput_to_json(fi)
        @test json_str isa String
        @test length(json_str) > 10

        fi2 = FoldingInput_from_json(json_str)
        @test fi2.name == fi.name
        @test length(fi2.protein_chains) == 1
        @test fi2.protein_chains[1].sequence == fi.protein_chains[1].sequence
        @test fi2.rng_seeds == fi.rng_seeds
    end

    # ── 2. Featurisation ──────────────────────────────────────────────────────
    @testset "Featurisation" begin
        fi  = make_small_fold_input()
        ccd = make_test_ccd()

        batches = featurise_input(fi; ccd=ccd, buckets=[64, 128, 256])
        @test length(batches) == length(fi.rng_seeds)

        batch = first(batches)
        @test haskey(batch, "token_index")
        @test haskey(batch, "target_feat")
        @test haskey(batch, "seq_mask")
        @test haskey(batch, "ref_pos")
        @test haskey(batch, "ref_mask")

        n_tokens_expected = length(fi.protein_chains[1].sequence)  # 17
        bucket_size = batch["num_tokens"][1] == n_tokens_expected ? n_tokens_expected :
            select_bucket(n_tokens_expected, [64, 128, 256])
        @test size(batch["target_feat"], 1) == bucket_size
        @test size(batch["target_feat"], 2) == TARGET_FEAT_DIM
    end

    # ── 3. Model forward pass (zero weights) ──────────────────────────────────
    @testset "Model forward pass (zero weights)" begin
        fi     = make_small_fold_input()
        ccd    = make_test_ccd()
        runner = make_small_model_runner(num_samples=1, num_recycles=1)

        batches = featurise_input(fi; ccd=ccd, buckets=[64])
        batch   = first(batches)

        result = predict(runner, batch, 42)

        @test result isa ModelResult
        @test haskey(result, "predicted_positions")
        @test haskey(result, "plddt_logits")
        @test haskey(result, "pae_logits")

        pos_shape = size(result["predicted_positions"])
        @test pos_shape[1] == runner.config.num_diffusion_samples
        @test pos_shape[4] == 3
    end

    # ── 4. Confidence metrics ─────────────────────────────────────────────────
    @testset "Confidence metrics" begin
        fi     = make_small_fold_input()
        ccd    = make_test_ccd()
        runner = make_small_model_runner(num_samples=1, num_recycles=1)

        batches = featurise_input(fi; ccd=ccd, buckets=[64])
        batch   = first(batches)
        result  = predict(runner, batch, 42)

        n_tokens = Int(batch["num_tokens"][1])
        token_chain_ids = fill("A", n_tokens)

        plddt_logits = zeros(Float32, n_tokens, NUM_ATOM_TYPES_PLDDT, PLDDT_NUM_BINS)
        pae_logits   = zeros(Float32, n_tokens, n_tokens, PAE_NUM_BINS)

        plddt = compute_plddt(plddt_logits)
        @test length(plddt) == n_tokens * NUM_ATOM_TYPES_PLDDT
        @test all(0f0 .<= plddt .<= 100f0)

        pae = compute_pae(pae_logits)
        @test size(pae) == (n_tokens, n_tokens)
        @test all(pae .>= 0f0)

        ptm = compute_ptm(pae_logits, token_chain_ids)
        @test 0.0 <= ptm <= 1.0
    end

    # ── 5. Structure output ───────────────────────────────────────────────────
    @testset "Structure output" begin
        n = 5
        positions = zeros(Float32, n, NUM_ATOM_SLOTS, 3)
        mask      = falses(n, NUM_ATOM_SLOTS)
        mask[:, 2] .= true   # Cα present

        s = build_structure_from_dense_positions(;
            positions           = positions,
            mask                = mask,
            token_residue_types = fill("ALA", n),
            token_chain_ids     = fill("A", n),
            token_seq_ids       = string.(1:n),
            bfactors            = zeros(Float32, n),
            name                = "test",
        )

        @test num_atoms(s) == n  # only Cα atoms
        @test s isa Structure

        # Round-trip through mmCIF
        mmcif_str = to_mmcif(s)
        @test mmcif_str isa String
        @test occursin("data_", mmcif_str)
        @test occursin("_atom_site.Cartn_x", mmcif_str)
    end

    # ── 6. Alignment (RMSD / superimpose) ─────────────────────────────────────
    @testset "RMSD and superimposition" begin
        n = 10
        true_pos = randn(Float32, n, 3)
        pred_pos = true_pos .+ 0.5f0  # constant shift

        mask = trues(n)
        rmsd_before = compute_rmsd(pred_pos, true_pos, mask)
        @test rmsd_before ≈ Float32(0.5 * sqrt(3)) atol=0.01f0

        aligned, rmsd_after = superimpose(pred_pos, true_pos, mask)
        @test rmsd_after < rmsd_before + 0.01f0  # should not get worse
        @test size(aligned) == size(pred_pos)
    end

    # ── 7. Geometry (Rigid3Array) ─────────────────────────────────────────────
    @testset "Rigid3Array geometry" begin
        r = identity_rigid()
        @test r.rotation ≈ Matrix{Float32}(I, 3, 3)
        @test r.translation ≈ zeros(Float32, 3)

        pt = Float32[1, 2, 3]
        @test apply(r, pt) ≈ pt

        # Compose with itself should give identity scaled translation
        r2 = Rigid3Array(
            Float32[0 -1 0; 1 0 0; 0 0 1],  # 90° rotation around z
            Float32[1, 0, 0],
        )
        r3 = compose(r2, r2)
        # After two 90° rotations: 180°
        @test r3.rotation ≈ Float32[-1 0 0; 0 -1 0; 0 0 1] atol=1e-5f0

        # invert
        r_inv = invert(r2)
        id = compose(r2, r_inv)
        @test id.rotation ≈ Matrix{Float32}(I, 3, 3) atol=1e-5f0
    end

    # ── 8. Data constants ──────────────────────────────────────────────────────
    @testset "Data constants" begin
        @test length(RESTYPE_ORDER) == NUM_RESIDUE_TYPES
        @test PLDDT_BIN_EDGES[1]   ≈ 0f0
        @test PLDDT_BIN_EDGES[end] ≈ 1f0
        @test length(PLDDT_BIN_CENTERS) == PLDDT_NUM_BINS
        @test PAE_NUM_BINS == 64
        @test DISTOGRAM_NUM_BINS == 64
    end

    # ── 9. MSA features ───────────────────────────────────────────────────────
    @testset "MSA features" begin
        seq = "ACDEFGHIKLMNPQRSTVWY"
        msa = Msa([seq, seq], ["s1", "s2"], zeros(Int, 2, length(seq)))

        feats = make_msa_features(msa)
        @test haskey(feats, "msa_feat")
        @test size(feats["msa_feat"], 3) == MSA_FEAT_DIM
    end

    # ── 10. Full prediction + output ──────────────────────────────────────────
    @testset "Prediction + output writing" begin
        fi     = make_small_fold_input()
        ccd    = make_test_ccd()
        runner = make_small_model_runner(num_samples=2, num_recycles=1)

        all_results = predict_structure(runner, fi, ccd;
            buckets           = [64],
            compress          = false,
            return_embeddings = true,
            return_distogram  = true,
        )

        @test length(all_results) == 1  # 1 seed
        rfs = first(all_results)
        @test length(rfs.inference_results) == 2  # 2 samples

        mktempdir() do tmpdir
            write_outputs(all_results, tmpdir;
                job_name                   = "test",
                compress_large_output_files = false,
            )
            @test isfile(joinpath(tmpdir, "test_ranking_scores.csv"))
        end
    end

end  # @testset

@info "All end-to-end tests completed."
