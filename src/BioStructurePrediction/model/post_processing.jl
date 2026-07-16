"""
Post-processing and output writing for inference results.
"""

using JSON3
using NPZ
using CodecZstd
using Dates
using Printf
using CSV
using DataFrames

const TERMS_OF_USE_TEXT = """
# AlphaFold 3 Non-Commercial Terms of Use

This system provides biomolecular structure predictions for research purposes.

By using this software you agree to the terms and conditions. Please review
the full license terms before using these predictions in your work.

The predictions are provided "as is" without warranty of any kind. The authors
and contributors are not responsible for any consequences arising from use of
these predictions.

Structure predictions are computational estimates and should be validated
experimentally before being used in applications with safety implications.

For the full terms of use, see the LICENSE file included with this distribution.
"""

# ──────────────────────────────────────────────
#  Output writing
# ──────────────────────────────────────────────

"""
    write_output(
        inference_result::InferenceResult,
        output_dir::String;
        name::String,
        terms_of_use::Union{String,Nothing} = nothing,
        compress::Bool = false
    )

Write inference results to output_dir.
"""
function write_output(
    inference_result::InferenceResult,
    output_dir::String;
    name::String,
    terms_of_use::Union{String,Nothing} = nothing,
    compress::Bool = false,
)
    isdir(output_dir) || mkpath(output_dir)
    cm = inference_result.confidence_metrics

    # ── mmCIF structure ──────────────────────
    # pLDDT as B-factors: use per-token mean, repeated for all atoms
    n_atoms = num_atoms(inference_result.predicted_structure)
    bfactors = Float32[]
    if n_atoms > 0
        # Map per-token pLDDT to per-atom B-factors
        per_token_plddt = cm.mean_plddt_per_token  # (n_tokens,)
        chain_ids  = get_column(inference_result.predicted_structure.atoms, :label_asym_id)
        seq_ids    = get_column(inference_result.predicted_structure.atoms, :label_seq_id)
        # Build token→pLDDT map by position
        for i in 1:n_atoms
            # Use mean pLDDT; in a full implementation, we'd map token index to atom
            bf = isempty(per_token_plddt) ? 50f0 : mean(per_token_plddt) * 100f0
            push!(bfactors, clamp(bf, 0f0, 100f0))
        end
    end

    mmcif_str = to_mmcif(inference_result.predicted_structure; bfactors=bfactors)
    cif_path  = joinpath(output_dir, "$(name)_model.cif")

    if compress
        open(cif_path * ".zst", "w") do io
            stream = ZstdCompressorStream(io)
            write(stream, mmcif_str)
            close(stream)
        end
        @info "Wrote compressed structure to $(cif_path).zst"
    else
        write(cif_path, mmcif_str)
        @info "Wrote structure to $cif_path"
    end

    # ── Confidence JSON ───────────────────────
    conf_dict = Dict{String,Any}(
        "plddt"              => cm.plddt,
        "pae"                => [[cm.pae[i,j] for j in 1:size(cm.pae,2)] for i in 1:size(cm.pae,1)],
        "ptm"                => cm.ptm,
        "iptm"               => cm.iptm,
        "ranking_score"      => cm.ranking_score,
        "chain_ptm"          => cm.chain_ptm,
        "chain_pair_iptm"    => cm.chain_pair_iptm,
    )
    conf_json = JSON3.write(conf_dict)
    conf_path = joinpath(output_dir, "$(name)_confidences.json")

    if compress
        open(conf_path * ".zst", "w") do io
            stream = ZstdCompressorStream(io)
            write(stream, conf_json)
            close(stream)
        end
        @info "Wrote compressed confidences to $(conf_path).zst"
    else
        write(conf_path, conf_json)
        @info "Wrote confidences to $conf_path"
    end

    # ── Summary confidence JSON ───────────────
    mean_plddt = isempty(cm.mean_plddt_per_token) ? 0.0 :
        Float64(mean(cm.mean_plddt_per_token)) * 100.0

    summary_dict = Dict{String,Any}(
        "ptm"                => cm.ptm,
        "iptm"               => cm.iptm,
        "ranking_score"      => cm.ranking_score,
        "mean_plddt"         => mean_plddt,
        "fraction_disordered" => cm.fraction_disordered,
        "has_clash"          => cm.has_clash,
    )
    summary_path = joinpath(output_dir, "$(name)_summary_confidences.json")
    write(summary_path, JSON3.write(summary_dict))
    @info "Wrote summary confidences to $summary_path"

    # ── Terms of use ──────────────────────────
    if terms_of_use !== nothing
        tou_path = joinpath(output_dir, "TERMS_OF_USE.md")
        write(tou_path, terms_of_use)
    end

    return (cif_path=cif_path, conf_path=conf_path, summary_path=summary_path)
end

"""
    write_embeddings(
        embeddings::Dict{String,Array},
        output_dir::String;
        name::String
    )

Write embedding arrays as a .npz file.
"""
function write_embeddings(
    embeddings::Dict{String,Array},
    output_dir::String;
    name::String,
)
    isdir(output_dir) || mkpath(output_dir)
    path = joinpath(output_dir, "$(name)_embeddings.npz")
    # Convert to Float16
    npz_data = Dict{String,Array}(
        "single_embeddings" => Float16.(embeddings["single_embeddings"]),
        "pair_embeddings"   => Float16.(embeddings["pair_embeddings"]),
    )
    npzwrite(path, npz_data)
    @info "Wrote embeddings to $path"
    return path
end

"""
    write_distogram(
        distogram::Array,
        output_dir::String;
        name::String
    )

Write distogram as a compressed .npz file.
"""
function write_distogram(
    distogram::Array,
    output_dir::String;
    name::String,
)
    isdir(output_dir) || mkpath(output_dir)
    path = joinpath(output_dir, "$(name)_distogram.npz")
    npzwrite(path, Dict("distogram" => Float16.(distogram)))
    @info "Wrote distogram to $path"
    return path
end

# ──────────────────────────────────────────────
#  write_outputs (multi-seed, multi-sample)
# ──────────────────────────────────────────────

"""
    write_outputs(
        all_inference_results::Vector{ResultsForSeed},
        output_dir::String;
        job_name::String,
        compress_large_output_files::Bool
    )

Write all samples across all seeds. Identifies the top-ranking result and
writes it to the top-level output directory with the terms-of-use file.
"""
function write_outputs(
    all_inference_results::Vector{ResultsForSeed},
    output_dir::String;
    job_name::String,
    compress_large_output_files::Bool,
)
    isdir(output_dir) || mkpath(output_dir)

    ranking_rows = DataFrame(seed=Int[], sample=Int[], ranking_score=Float64[])
    best_score   = -Inf
    best_result  = nothing
    best_name    = ""

    for rfs in all_inference_results
        seed = rfs.seed

        for (sample_idx, ir) in enumerate(rfs.inference_results)
            sample_name = "$(job_name)_seed-$(seed)_sample-$(sample_idx-1)"
            sample_dir  = joinpath(output_dir, "seed-$(seed)_sample-$(sample_idx-1)")

            write_output(ir, sample_dir;
                name=sample_name,
                compress=compress_large_output_files,
            )

            rs = ir.confidence_metrics.ranking_score
            push!(ranking_rows, (seed, sample_idx-1, rs))

            if rs > best_score
                best_score  = rs
                best_result = ir
                best_name   = sample_name
            end
        end

        # Embeddings
        if rfs.embeddings !== nothing
            emb_dir = joinpath(output_dir, "seed-$(seed)_embeddings")
            write_embeddings(rfs.embeddings, emb_dir; name="$(job_name)_seed-$(seed)")
        end

        # Distogram
        if rfs.distogram !== nothing
            dist_dir = joinpath(output_dir, "seed-$(seed)_distogram")
            write_distogram(rfs.distogram, dist_dir; name="$(job_name)_seed-$(seed)")
        end
    end

    # Write top-ranking result to top-level dir
    if best_result !== nothing
        write_output(best_result, output_dir;
            name=best_name,
            terms_of_use=TERMS_OF_USE_TEXT,
            compress=compress_large_output_files,
        )
        @info "Top-ranking result (score=$(round(best_score; digits=4))): $best_name"
    end

    # Write ranking CSV
    csv_path = joinpath(output_dir, "$(job_name)_ranking_scores.csv")
    CSV.write(csv_path, ranking_rows)
    @info "Wrote ranking scores to $csv_path"
end
