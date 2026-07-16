"""
Folding input data model and JSON parsing for the biomolecular structure prediction pipeline.
"""

using JSON3
using Dates

# ──────────────────────────────────────────────
#  Entity structs
# ──────────────────────────────────────────────

struct ProteinModification
    ptm_type::String
    ptm_position::Int
end

struct RnaModification
    modification_type::String
    base_position::Int
end

struct DnaModification
    modification_type::String
    base_position::Int
end

struct TemplateHitInput
    mmcif::String
    query_indices::Vector{Int}
    template_indices::Vector{Int}
end

struct ProteinChain
    ids::Vector{String}            # one or more chain IDs
    sequence::String
    modifications::Vector{ProteinModification}
    unpaired_msa::Union{String,Nothing}
    paired_msa::Union{String,Nothing}
    templates::Union{Vector{TemplateHitInput},Nothing}
end

struct RnaChain
    ids::Vector{String}
    sequence::String
    modifications::Vector{RnaModification}
    unpaired_msa::Union{String,Nothing}
    paired_msa::Union{String,Nothing}
end

struct DnaChain
    ids::Vector{String}
    sequence::String
    modifications::Vector{DnaModification}
    unpaired_msa::Union{String,Nothing}
end

struct LigandEntity
    ids::Vector{String}
    ccd_codes::Vector{String}
    smiles::Union{String,Nothing}
end

struct BondedAtomPair
    chain1::String
    res1::Int
    atom1::String
    chain2::String
    res2::Int
    atom2::String
end

# ──────────────────────────────────────────────
#  Top-level FoldingInput
# ──────────────────────────────────────────────

"""
    FoldingInput

Complete description of a biomolecular structure prediction job.
"""
struct FoldingInput
    name::String
    rng_seeds::Vector{Int}
    protein_chains::Vector{ProteinChain}
    rna_chains::Vector{RnaChain}
    dna_chains::Vector{DnaChain}
    ligands::Vector{LigandEntity}
    bonded_atom_pairs::Vector{BondedAtomPair}
    user_ccd::Union{String,Nothing}
    dialect::String
    version::Int
end

# ──────────────────────────────────────────────
#  Accessor helpers
# ──────────────────────────────────────────────

"""
    chains(fi::FoldingInput)

Return all chains (proteins, RNA, DNA, ligands) as a flat vector of Any.
"""
function chains(fi::FoldingInput)
    return vcat(fi.protein_chains, fi.rna_chains, fi.dna_chains, fi.ligands)
end

"""
    sanitised_name(fi::FoldingInput) -> String

Replace characters not in [A-Za-z0-9_-] with underscores.
"""
function sanitised_name(fi::FoldingInput)::String
    return replace(fi.name, r"[^A-Za-z0-9_\-]" => "_")
end

"""
    with_multiple_seeds(fi::FoldingInput, n::Int) -> FoldingInput

Generate n sequential seeds starting from fi.rng_seeds[1].
Requires exactly one seed in fi.
"""
function with_multiple_seeds(fi::FoldingInput, n::Int)::FoldingInput
    if length(fi.rng_seeds) != 1
        error("--num_seeds requires exactly one seed in modelSeeds, got $(length(fi.rng_seeds))")
    end
    base = fi.rng_seeds[1]
    seeds = collect(base:base+n-1)
    return FoldingInput(
        fi.name, seeds, fi.protein_chains, fi.rna_chains, fi.dna_chains,
        fi.ligands, fi.bonded_atom_pairs, fi.user_ccd, fi.dialect, fi.version
    )
end

# ──────────────────────────────────────────────
#  JSON parsing helpers
# ──────────────────────────────────────────────

function _parse_ids(id_field)::Vector{String}
    if id_field isa AbstractString
        return [String(id_field)]
    elseif id_field isa AbstractArray
        return String[String(x) for x in id_field]
    else
        error("Invalid 'id' field type: $(typeof(id_field))")
    end
end

function _parse_protein_chain(d::AbstractDict)::ProteinChain
    ids = _parse_ids(d["id"])
    seq = String(d["sequence"])
    mods = ProteinModification[]
    if haskey(d, "modifications") && d["modifications"] !== nothing
        for m in d["modifications"]
            push!(mods, ProteinModification(String(m["ptmType"]), Int(m["ptmPosition"])))
        end
    end
    unpaired_msa = haskey(d, "unpairedMsa") && d["unpairedMsa"] !== nothing ? String(d["unpairedMsa"]) : nothing
    paired_msa   = haskey(d, "pairedMsa")   && d["pairedMsa"]   !== nothing ? String(d["pairedMsa"])   : nothing
    templates = nothing
    if haskey(d, "templates") && d["templates"] !== nothing
        templates = TemplateHitInput[]
        for t in d["templates"]
            push!(templates, TemplateHitInput(
                String(t["mmcif"]),
                Int[Int(x) for x in t["queryIndices"]],
                Int[Int(x) for x in t["templateIndices"]],
            ))
        end
    end
    return ProteinChain(ids, seq, mods, unpaired_msa, paired_msa, templates)
end

function _parse_rna_chain(d::AbstractDict)::RnaChain
    ids = _parse_ids(d["id"])
    seq = String(d["sequence"])
    mods = RnaModification[]
    if haskey(d, "modifications") && d["modifications"] !== nothing
        for m in d["modifications"]
            push!(mods, RnaModification(String(m["modificationType"]), Int(m["basePosition"])))
        end
    end
    unpaired_msa = haskey(d, "unpairedMsa") && d["unpairedMsa"] !== nothing ? String(d["unpairedMsa"]) : nothing
    paired_msa   = haskey(d, "pairedMsa")   && d["pairedMsa"]   !== nothing ? String(d["pairedMsa"])   : nothing
    return RnaChain(ids, seq, mods, unpaired_msa, paired_msa)
end

function _parse_dna_chain(d::AbstractDict)::DnaChain
    ids = _parse_ids(d["id"])
    seq = String(d["sequence"])
    mods = DnaModification[]
    if haskey(d, "modifications") && d["modifications"] !== nothing
        for m in d["modifications"]
            push!(mods, DnaModification(String(m["modificationType"]), Int(m["basePosition"])))
        end
    end
    unpaired_msa = haskey(d, "unpairedMsa") && d["unpairedMsa"] !== nothing ? String(d["unpairedMsa"]) : nothing
    return DnaChain(ids, seq, mods, unpaired_msa)
end

function _parse_ligand(d::AbstractDict)::LigandEntity
    ids      = _parse_ids(d["id"])
    ccd_codes = String[String(c) for c in get(d, "ccdCodes", [])]
    smiles   = haskey(d, "smiles") && d["smiles"] !== nothing ? String(d["smiles"]) : nothing
    return LigandEntity(ids, ccd_codes, smiles)
end

function _parse_bonded_atom_pairs(arr)::Vector{BondedAtomPair}
    pairs = BondedAtomPair[]
    for p in arr
        push!(pairs, BondedAtomPair(
            String(p[1]), Int(p[2]), String(p[3]),
            String(p[4]), Int(p[5]), String(p[6]),
        ))
    end
    return pairs
end

# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

"""
    FoldingInput_from_json(json_str::String) -> FoldingInput

Parse a JSON string into a FoldingInput struct.
"""
function FoldingInput_from_json(json_str::String)::FoldingInput
    d = JSON3.read(json_str)

    name    = String(d["name"])
    seeds   = Int[Int(s) for s in d["modelSeeds"]]
    dialect = String(d["dialect"])
    version = Int(d["version"])

    if dialect != "alphafold3"
        error("Unsupported dialect '$dialect'. Only 'alphafold3' is supported.")
    end
    if !(1 <= version <= 4)
        error("Unsupported version $version. Must be 1–4.")
    end

    protein_chains = ProteinChain[]
    rna_chains     = RnaChain[]
    dna_chains     = DnaChain[]
    ligands        = LigandEntity[]

    for entity in d["sequences"]
        if haskey(entity, "protein")
            push!(protein_chains, _parse_protein_chain(entity["protein"]))
        elseif haskey(entity, "rna")
            push!(rna_chains, _parse_rna_chain(entity["rna"]))
        elseif haskey(entity, "dna")
            push!(dna_chains, _parse_dna_chain(entity["dna"]))
        elseif haskey(entity, "ligand")
            push!(ligands, _parse_ligand(entity["ligand"]))
        else
            @warn "Unknown entity type in sequences: $(keys(entity))"
        end
    end

    bonded_pairs = BondedAtomPair[]
    if haskey(d, "bondedAtomPairs") && d["bondedAtomPairs"] !== nothing
        bonded_pairs = _parse_bonded_atom_pairs(d["bondedAtomPairs"])
    end

    user_ccd = nothing
    if haskey(d, "userCCD") && d["userCCD"] !== nothing
        user_ccd = String(d["userCCD"])
    elseif haskey(d, "userCCDPath") && d["userCCDPath"] !== nothing
        path = String(d["userCCDPath"])
        isfile(path) || error("userCCDPath '$path' does not exist")
        user_ccd = read(path, String)
    end

    return FoldingInput(name, seeds, protein_chains, rna_chains, dna_chains, ligands,
                        bonded_pairs, user_ccd, dialect, version)
end

"""
    FoldingInput_to_json(fi::FoldingInput) -> String

Serialise a FoldingInput struct to a JSON string.
"""
function FoldingInput_to_json(fi::FoldingInput)::String
    sequences = Any[]
    for chain in fi.protein_chains
        id_field = length(chain.ids) == 1 ? chain.ids[1] : chain.ids
        mods = [Dict("ptmType" => m.ptm_type, "ptmPosition" => m.ptm_position) for m in chain.modifications]
        templates_out = if chain.templates === nothing
            nothing
        else
            [Dict("mmcif" => t.mmcif, "queryIndices" => t.query_indices, "templateIndices" => t.template_indices)
             for t in chain.templates]
        end
        push!(sequences, Dict("protein" => Dict(
            "id" => id_field,
            "sequence" => chain.sequence,
            "modifications" => mods,
            "unpairedMsa" => chain.unpaired_msa,
            "pairedMsa"   => chain.paired_msa,
            "templates"   => templates_out,
        )))
    end
    for chain in fi.rna_chains
        id_field = length(chain.ids) == 1 ? chain.ids[1] : chain.ids
        mods = [Dict("modificationType" => m.modification_type, "basePosition" => m.base_position) for m in chain.modifications]
        push!(sequences, Dict("rna" => Dict(
            "id" => id_field,
            "sequence" => chain.sequence,
            "modifications" => mods,
            "unpairedMsa" => chain.unpaired_msa,
            "pairedMsa"   => chain.paired_msa,
        )))
    end
    for chain in fi.dna_chains
        id_field = length(chain.ids) == 1 ? chain.ids[1] : chain.ids
        mods = [Dict("modificationType" => m.modification_type, "basePosition" => m.base_position) for m in chain.modifications]
        push!(sequences, Dict("dna" => Dict(
            "id" => id_field,
            "sequence" => chain.sequence,
            "modifications" => mods,
            "unpairedMsa" => chain.unpaired_msa,
        )))
    end
    for lig in fi.ligands
        id_field = length(lig.ids) == 1 ? lig.ids[1] : lig.ids
        push!(sequences, Dict("ligand" => Dict(
            "id" => id_field,
            "ccdCodes" => lig.ccd_codes,
            "smiles" => lig.smiles,
        )))
    end

    bonded_pairs_out = [[p.chain1, p.res1, p.atom1, p.chain2, p.res2, p.atom2] for p in fi.bonded_atom_pairs]

    obj = Dict(
        "name"     => fi.name,
        "modelSeeds" => fi.rng_seeds,
        "dialect"  => fi.dialect,
        "version"  => fi.version,
        "sequences" => sequences,
        "bondedAtomPairs" => isempty(bonded_pairs_out) ? nothing : bonded_pairs_out,
        "userCCD"  => fi.user_ccd,
    )
    return JSON3.write(obj)
end

"""
    load_fold_inputs_from_path(path::String) -> Vector{FoldingInput}

Load one or more FoldingInput objects from a JSON file.
The file may contain a single object or a JSON array of objects.
"""
function load_fold_inputs_from_path(path::String)::Vector{FoldingInput}
    isfile(path) || error("Input file not found: $path")
    json_str = read(path, String)
    parsed = JSON3.read(json_str)
    if parsed isa AbstractArray
        return FoldingInput[FoldingInput_from_json(JSON3.write(obj)) for obj in parsed]
    else
        return [FoldingInput_from_json(json_str)]
    end
end

"""
    load_fold_inputs_from_dir(dir::String) -> Vector{FoldingInput}

Load FoldingInput objects from all .json files in dir (sorted by filename).
"""
function load_fold_inputs_from_dir(dir::String)::Vector{FoldingInput}
    isdir(dir) || error("Input directory not found: $dir")
    json_files = sort(filter(f -> endswith(f, ".json"), readdir(dir; join=true)))
    isempty(json_files) && error("No .json files found in directory: $dir")
    result = FoldingInput[]
    for path in json_files
        append!(result, load_fold_inputs_from_path(path))
    end
    return result
end

"""
    write_fold_input_json(fi::FoldingInput, output_dir::String)

Write the FoldingInput to {output_dir}/{sanitised_name}_data.json.
"""
function write_fold_input_json(fi::FoldingInput, output_dir::String)
    isdir(output_dir) || mkpath(output_dir)
    path = joinpath(output_dir, sanitised_name(fi) * "_data.json")
    write(path, FoldingInput_to_json(fi))
    @info "Wrote input JSON to $path"
    return path
end
