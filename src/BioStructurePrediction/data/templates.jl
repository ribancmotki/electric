"""
Template search and processing for protein structure prediction.
"""

using Dates

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

"""
    TemplateSearchConfig

Configuration for template search.
"""
struct TemplateSearchConfig
    max_templates::Int                     # Default 4
    max_template_date::Date
    pdb_database_path::String
    seqres_database_path::String
    hmmsearch_binary_path::String
    hmmbuild_binary_path::String
    hmmalign_binary_path::String
end

"""
    TemplateHit

A matched template for a query protein chain.
"""
struct TemplateHit
    pdb_id::String
    chain_id::String
    mmcif::String
    query_indices::Vector{Int}
    template_indices::Vector{Int}
    evalue::Float64
end

# ──────────────────────────────────────────────
#  Template search
# ──────────────────────────────────────────────

"""
    search_templates(
        query_sequence::String,
        uniref90_msa::Msa,
        config::TemplateSearchConfig
    ) -> Vector{TemplateHit}

Search for structural templates for a protein chain.
"""
function search_templates(
    query_sequence::String,
    uniref90_msa::Msa,
    config::TemplateSearchConfig
)::Vector{TemplateHit}
    # Check required binaries
    for (name, path) in [("hmmbuild", config.hmmbuild_binary_path),
                          ("hmmsearch", config.hmmsearch_binary_path)]
        if isempty(path) || !isfile(path)
            @warn "$name binary not found at '$path'; skipping template search"
            return TemplateHit[]
        end
    end
    if !isfile(config.seqres_database_path)
        @warn "Seqres database not found: $(config.seqres_database_path); skipping template search"
        return TemplateHit[]
    end
    if !isdir(config.pdb_database_path)
        @warn "PDB mmCIF directory not found: $(config.pdb_database_path); skipping template search"
        return TemplateHit[]
    end

    tmp_dir = mktempdir()
    hits    = TemplateHit[]
    try
        # Step 1: Write MSA to temp file in Stockholm format for hmmbuild
        msa_file = joinpath(tmp_dir, "query.sto")
        open(msa_file, "w") do io
            println(io, "# STOCKHOLM 1.0")
            for (i, (desc, seq)) in enumerate(zip(uniref90_msa.descriptions, uniref90_msa.sequences))
                println(io, "$desc\t$seq")
            end
            println(io, "//")
        end

        # Step 2: Build HMM from MSA
        hmm_file = joinpath(tmp_dir, "query.hmm")
        run_hmmbuild(config.hmmbuild_binary_path, msa_file, hmm_file)

        # Step 3: Search seqres with hmmsearch
        tblout_file = joinpath(tmp_dir, "hits.tblout")
        hmmsearch_stdout = run_hmmsearch(config.hmmsearch_binary_path, hmm_file,
                                          config.seqres_database_path, tblout_file)

        # Step 4: Parse and filter hits
        raw_hits = parse_hmmsearch_output(hmmsearch_stdout, config.max_template_date)

        # Step 5: Retrieve mmCIF and realign
        store = PdbStructureStore(config.pdb_database_path)
        n_retrieved = 0
        for hit in raw_hits
            n_retrieved >= config.max_templates && break

            # Retrieve mmCIF
            mmcif_str = get_structure(store, hit.pdb_id)
            mmcif_str === nothing && continue

            # Filter by date
            release_date = read_structure_release_date(mmcif_str)
            if release_date !== nothing && release_date > config.max_template_date
                continue
            end

            # Get template sequence from mmCIF
            template_struct = parse_structure_from_mmcif_string(mmcif_str)
            template_seq    = extract_sequence_for_chain(template_struct, hit.chain_id)
            isempty(template_seq) && continue

            # Realign
            query_indices, template_indices = realign_template(
                query_sequence,
                template_seq,
                config.hmmalign_binary_path;
                hmm_profile_path = hmm_file,
            )

            isempty(query_indices) && continue

            push!(hits, TemplateHit(
                hit.pdb_id, hit.chain_id, mmcif_str,
                query_indices, template_indices, hit.evalue,
            ))
            n_retrieved += 1
        end

    catch e
        @error "Template search failed" exception=(e, catch_backtrace())
    finally
        rm(tmp_dir; recursive=true, force=true)
    end

    return hits
end

# ──────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────

"""
    run_hmmbuild(binary::String, msa_file::String, hmm_file::String)

Run hmmbuild to build an HMM from an MSA.
"""
function run_hmmbuild(binary::String, msa_file::String, hmm_file::String)
    cmd = `$binary --amino $hmm_file $msa_file`
    run(pipeline(cmd; stdout=devnull, stderr=devnull))
end

"""
    run_hmmsearch(binary::String, hmm_file::String, db_file::String, tblout_file::String) -> String

Run hmmsearch and return the stdout.
"""
function run_hmmsearch(binary::String, hmm_file::String, db_file::String, tblout_file::String)::String
    cmd = `$binary --noali --cpu 4 --tblout $tblout_file $hmm_file $db_file`
    return read(pipeline(cmd; stderr=devnull), String)
end

"""
    extract_sequence_for_chain(s::Structure, chain_id::String) -> String

Extract the one-letter sequence for the given chain from a Structure.
"""
function extract_sequence_for_chain(s::Structure, chain_id::String)::String
    chain_ids = get_column(s.atoms, :label_asym_id)
    comp_ids  = get_column(s.atoms, :label_comp_id)
    seq_ids   = get_column(s.atoms, :label_seq_id)

    seen_residues = Set{String}()
    ordered_residues = String[]
    for i in 1:num_atoms(s)
        chain_ids[i] == chain_id || continue
        sid = string(seq_ids[i])
        sid ∈ seen_residues && continue
        push!(seen_residues, sid)
        push!(ordered_residues, comp_ids[i])
    end

    buf = IOBuffer()
    for comp in ordered_residues
        std_comp = standardise_residue_name(comp)
        one = get(THREE_LETTER_TO_ONE, std_comp, 'X')
        print(buf, one)
    end
    return String(take!(buf))
end

"""
    template_hits_to_input(hits::Vector{TemplateHit}) -> Vector{TemplateHitInput}

Convert TemplateHit objects to TemplateHitInput for storage in FoldingInput.
"""
function template_hits_to_input(hits::Vector{TemplateHit})::Vector{TemplateHitInput}
    return [TemplateHitInput(h.mmcif, h.query_indices, h.template_indices) for h in hits]
end
