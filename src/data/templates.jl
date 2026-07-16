"""
templates.jl — Template search and featurization.
"""

using Dates
using Logging

# ──────────────────────────────────────────────────────────────────────────────
# Template hit types
# ──────────────────────────────────────────────────────────────────────────────

"""
    TemplateHit

A single template search hit.
"""
struct TemplateHit
    index::Int
    name::String
    aligned_cols::Int
    seq_id::Float64
    score::Float64
    e_value::Float64
    query_alignment::String
    hit_alignment::String
    hit_sequence::String
end

function Base.show(io::IO, h::TemplateHit)
    print(io, "TemplateHit($(h.name), e=$(h.e_value), score=$(h.score))")
end

# ──────────────────────────────────────────────────────────────────────────────
# Templates struct
# ──────────────────────────────────────────────────────────────────────────────

"""
    Templates

Collection of template hits with featurization capability.
"""
struct Templates
    query_sequence::String
    hits::Vector{TemplateHit}
    max_template_date::Date
    structure_store::Any   # PdbStructureStore
    filter_config::TemplateFilterConfig
    chain_poly_type::String
end

function Base.show(io::IO, t::Templates)
    print(io, "Templates($(length(t.hits)) hits, chain_type=$(t.chain_poly_type))")
end

function num_hits(t::Templates)::Int
    return length(t.hits)
end

# ──────────────────────────────────────────────────────────────────────────────
# Template search
# ──────────────────────────────────────────────────────────────────────────────

"""
    templates_from_seq_and_a3m(query_sequence, msa_a3m, max_template_date,
                                database_path, hmmsearch_config,
                                max_a3m_query_sequences, chain_poly_type,
                                structure_store, filter_config) -> Templates

Search for templates using hmmsearch.
"""
function templates_from_seq_and_a3m(
    query_sequence::String,
    msa_a3m::String,
    max_template_date::Date,
    database_path::String,
    hmmsearch_config::HmmsearchConfig,
    max_a3m_query_sequences::Int,
    chain_poly_type::String,
    structure_store,
    filter_config::TemplateFilterConfig,
)::Templates
    hits = TemplateHit[]

    if !isfile(database_path)
        @warn "Template database not found: $database_path"
        return Templates(query_sequence, hits, max_template_date,
                         structure_store, filter_config, chain_poly_type)
    end

    try
        # Build HMM from MSA
        hmmbuild_cfg = HmmbuildConfig(
            binary_path=hmmsearch_config.hmmbuild_binary_path, n_cpu=4)
        hmm = run_hmmbuild(hmmbuild_cfg, msa_a3m)

        # Run hmmsearch
        tblout = run_hmmsearch(hmmsearch_config, hmm, database_path)
        raw_hits = parse_hmmsearch_output(tblout)

        # Filter and convert
        for (i, h) in enumerate(raw_hits)
            e_val = h["e_value"]
            e_val > filter_config.max_subsequence_ratio * 1000 && continue  # placeholder filter

            hit = TemplateHit(
                i,
                String(h["target_name"]),
                0,   # aligned_cols
                0.0, # seq_id
                h["score"],
                e_val,
                "",  # query_alignment (fill in later)
                "",  # hit_alignment
                "",  # hit_sequence
            )
            push!(hits, hit)
        end

        # Apply max_hits filter
        if length(hits) > filter_config.max_hits
            hits = hits[1:filter_config.max_hits]
        end

    catch e
        @warn "Template search failed: $e"
    end

    return Templates(query_sequence, hits, max_template_date,
                     structure_store, filter_config, chain_poly_type)
end

# ──────────────────────────────────────────────────────────────────────────────
# Template featurization
# ──────────────────────────────────────────────────────────────────────────────

"""
    get_polymer_features(chain_struct::Structure, chain_poly_type::String,
                         query_seq_length::Int,
                         query_to_hit_map::Dict{Int,Int}) -> Dict{String,Array}

Extract template features from a structure.
Returns:
- "template_aatype": Int32(num_res,)
- "template_atom_positions": Float32(num_res, 24, 3)
- "template_atom_mask": Bool(num_res, 24)
"""
function get_polymer_features(
    chain_struct::Structure,
    chain_poly_type::String,
    query_seq_length::Int,
    query_to_hit_map::Dict{Int,Int},
)::Dict{String,Array}
    n_template_atoms = 24  # backbone + sidechain atoms

    aatype          = zeros(Int32, query_seq_length)
    atom_positions  = zeros(Float32, query_seq_length, n_template_atoms, 3)
    atom_mask       = zeros(Bool, query_seq_length, n_template_atoms)

    # Get residues from template structure
    res_map = Dict{Int,Dict{String,Vector{Float32}}}()  # res_id → atom_name → coords
    for i in 1:length(chain_struct)
        rid = chain_struct.res_id[i]
        haskey(res_map, rid) || (res_map[rid] = Dict{String,Vector{Float32}}())
        res_map[rid][chain_struct.atom_name[i]] = [
            chain_struct.atom_x[i], chain_struct.atom_y[i], chain_struct.atom_z[i]
        ]
    end

    # Get residue names
    res_name_map = Dict{Int,String}()
    for i in 1:length(chain_struct)
        res_name_map[chain_struct.res_id[i]] = chain_struct.res_name[i]
    end

    # Map query positions to template
    template_res_ids = sort(collect(keys(res_map)))

    for (q_idx, t_idx) in query_to_hit_map
        1 <= q_idx <= query_seq_length || continue
        t_idx <= length(template_res_ids) || continue

        t_res_id = template_res_ids[t_idx]
        rn = get(res_name_map, t_res_id, "UNK")

        # Encode residue type
        aatype[q_idx] = Int32(get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP,
                                  letters_three_to_one(rn), 0))

        # Fill in atom positions
        atom_coords = get(res_map, t_res_id, Dict{String,Vector{Float32}}())
        for (atom_k, atom_name) in enumerate(ATOM_ORDER)
            atom_k > n_template_atoms && break
            if haskey(atom_coords, atom_name)
                coords = atom_coords[atom_name]
                atom_positions[q_idx, atom_k, :] = coords
                atom_mask[q_idx, atom_k] = true
            end
        end
    end

    return Dict{String,Array}(
        "template_aatype"          => aatype,
        "template_atom_positions"  => atom_positions,
        "template_atom_mask"       => atom_mask,
    )
end

"""
    package_template_features(hit_features::Vector{Dict}, include_ligand_features::Bool)
        -> Dict{String,Array}

Stack per-hit template feature dicts into a single batched dict.
"""
function package_template_features(hit_features::Vector{Dict},
                                   include_ligand_features::Bool)::Dict{String,Array}
    isempty(hit_features) && return Dict{String,Array}()

    result = Dict{String,Array}()
    for key in keys(hit_features[1])
        arrays = [f[key] for f in hit_features if haskey(f, key)]
        isempty(arrays) && continue
        result[key] = cat(arrays...; dims=1)
    end
    return result
end

"""
    get_hits_with_structures(t::Templates) -> Vector{Tuple{TemplateHit,Union{Structure,Nothing}}}

Load structures for all template hits.
"""
function get_hits_with_structures(t::Templates)
    results = Tuple{TemplateHit,Union{Structure,Nothing}}[]
    for hit in t.hits
        # Extract PDB ID from hit name (e.g., "4hhb_A" → "4hhb")
        pdb_id = split(lowercase(hit.name), "_")[1]
        struct_data = if t.structure_store !== nothing
            try
                get_structure(t.structure_store, pdb_id)
            catch e
                @warn "Failed to load template structure $pdb_id: $e"
                nothing
            end
        else
            nothing
        end
        push!(results, (hit, struct_data))
    end
    return results
end
