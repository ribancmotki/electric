"""
folding_input.jl — Input data structures for the structure prediction pipeline.
"""

using JSON3
using Dates
using Logging

# Forward declaration — CCD is defined in constants/chemical_components.jl
# and will be available at runtime within the module.

# ──────────────────────────────────────────────────────────────────────────────
# Template
# ──────────────────────────────────────────────────────────────────────────────

"""
    Template

A structural template for protein chain featurization.
- `mmcif`: full mmCIF string of the template structure (single protein chain).
- `query_to_template_map`: 1-based query residue index → 1-based template residue index.
"""
struct Template
    mmcif::String
    query_to_template_map::Dict{Int,Int}
end

function Base.hash(t::Template, h::UInt)
    return hash(t.mmcif, hash(sort(collect(t.query_to_template_map)), h))
end

function Base.:(==)(a::Template, b::Template)
    return a.mmcif == b.mmcif && a.query_to_template_map == b.query_to_template_map
end

function Base.show(io::IO, t::Template)
    print(io, "Template($(length(t.mmcif)) mmCIF bytes, $(length(t.query_to_template_map)) mapped residues)")
end

# ──────────────────────────────────────────────────────────────────────────────
# Modification (PTM / nucleotide modification)
# ──────────────────────────────────────────────────────────────────────────────

"""
ProteinModification: post-translational modification at a specific residue.
"""
struct ProteinModification
    ccd_id::String      # CCD code of the modifying group
    position::Int       # 1-based residue index in the protein sequence
end

# ──────────────────────────────────────────────────────────────────────────────
# ProteinChain
# ──────────────────────────────────────────────────────────────────────────────

"""
    ProteinChain

A protein chain input for structure prediction.
"""
struct ProteinChain
    id::String
    sequence::String
    ptms::Vector{Tuple{String,Int}}
    description::Union{String,Nothing}
    paired_msa::Union{String,Nothing}
    unpaired_msa::Union{String,Nothing}
    templates::Union{Vector{Template},Nothing}
end

function ProteinChain(;
    id::String,
    sequence::String,
    ptms::Vector{Tuple{String,Int}} = Tuple{String,Int}[],
    description::Union{String,Nothing} = nothing,
    paired_msa::Union{String,Nothing} = nothing,
    unpaired_msa::Union{String,Nothing} = nothing,
    templates::Union{Vector{Template},Nothing} = nothing,
)
    # Validate
    @assert all(isletter, sequence) "ProteinChain sequence must contain only letters, got: $sequence"
    for (ccd, idx) in ptms
        @assert 1 <= idx <= length(sequence) "PTM index $idx out of range [1,$(length(sequence))]"
        @assert !startswith(ccd, "CCD_") "PTM CCD code must not start with 'CCD_', got: $ccd"
    end
    @assert all(isuppercase, id) "Chain ID must be uppercase, got: $id"
    return ProteinChain(id, sequence, ptms, description, paired_msa, unpaired_msa, templates)
end

function get_sequence(chain::ProteinChain)::String
    return chain.sequence
end

"""
    to_ccd_sequence(chain::ProteinChain) -> Vector{String}

Convert protein chain to a vector of CCD residue codes, applying PTMs.
"""
function to_ccd_sequence(chain::ProteinChain)::Vector{String}
    residues = protein_sequence_to_residues(chain.sequence)
    for (ccd, idx) in chain.ptms
        residues[idx] = ccd
    end
    return residues
end

"""
    fill_missing_fields(chain::ProteinChain) -> ProteinChain

Fill nothing MSA/template fields with empty defaults.
"""
function fill_missing_fields(chain::ProteinChain)::ProteinChain
    paired   = chain.paired_msa   === nothing ? ">query\n$(chain.sequence)\n" : chain.paired_msa
    unpaired = chain.unpaired_msa === nothing ? ">query\n$(chain.sequence)\n" : chain.unpaired_msa
    templates = chain.templates   === nothing ? Template[] : chain.templates
    return ProteinChain(chain.id, chain.sequence, chain.ptms, chain.description,
                        paired, unpaired, templates)
end

function hash_without_id(chain::ProteinChain)::UInt
    return hash((chain.sequence, chain.ptms))
end

function Base.show(io::IO, c::ProteinChain)
    print(io, "ProteinChain(id=$(c.id), len=$(length(c.sequence)), ptms=$(length(c.ptms)))")
end

function from_dict_protein(d::Dict)::ProteinChain
    protein = get(d, "protein", d)
    id   = String(get(protein, "id", "A"))
    seq  = String(get(protein, "sequence", ""))
    mods = get(protein, "modifications", Any[])
    ptms = Tuple{String,Int}[
        (String(m["ptmType"]), Int(m["ptmPosition"])) for m in mods
    ]
    upaired_msa = _read_msa_field(protein, "unpairedMsa", "unpairedMsaPath")
    paired_msa  = _read_msa_field(protein, "pairedMsa",   "pairedMsaPath")
    templates_raw = get(protein, "templates", nothing)
    templates = if templates_raw !== nothing
        [_dict_to_template(t) for t in templates_raw]
    else
        nothing
    end
    desc = get(protein, "description", nothing)
    desc = desc isa String ? desc : nothing
    return ProteinChain(
        id=id, sequence=seq, ptms=ptms, description=desc,
        paired_msa=paired_msa, unpaired_msa=upaired_msa, templates=templates,
    )
end

function to_dict(chain::ProteinChain)::Dict
    d = Dict{String,Any}(
        "id"       => chain.id,
        "sequence" => chain.sequence,
        "modifications" => [Dict("ptmType"=>c,"ptmPosition"=>i) for (c,i) in chain.ptms],
    )
    chain.paired_msa   !== nothing && (d["pairedMsa"]   = chain.paired_msa)
    chain.unpaired_msa !== nothing && (d["unpairedMsa"] = chain.unpaired_msa)
    chain.templates    !== nothing && (d["templates"]   = [_template_to_dict(t) for t in chain.templates])
    chain.description  !== nothing && (d["description"] = chain.description)
    return Dict("protein" => d)
end

# ──────────────────────────────────────────────────────────────────────────────
# RnaChain
# ──────────────────────────────────────────────────────────────────────────────

struct RnaChain
    id::String
    sequence::String
    modifications::Vector{Tuple{String,Int}}
    description::Union{String,Nothing}
    unpaired_msa::Union{String,Nothing}
end

function RnaChain(;
    id::String,
    sequence::String,
    modifications::Vector{Tuple{String,Int}} = Tuple{String,Int}[],
    description::Union{String,Nothing} = nothing,
    unpaired_msa::Union{String,Nothing} = nothing,
)
    @assert all(c -> c in "ACGUNacgun", sequence) "RnaChain sequence must be ACGUN, got non-RNA char"
    @assert all(isuppercase, id) "Chain ID must be uppercase, got: $id"
    for (ccd, idx) in modifications
        @assert 1 <= idx <= length(sequence) "Modification index $idx out of range"
    end
    return RnaChain(id, uppercase(sequence), modifications, description, unpaired_msa)
end

function to_ccd_sequence(chain::RnaChain)::Vector{String}
    residues = rna_sequence_to_residues(chain.sequence)
    for (ccd, idx) in chain.modifications
        residues[idx] = ccd
    end
    return residues
end

function fill_missing_fields(chain::RnaChain)::RnaChain
    unpaired = chain.unpaired_msa === nothing ? ">query\n$(chain.sequence)\n" : chain.unpaired_msa
    return RnaChain(chain.id, chain.sequence, chain.modifications, chain.description, unpaired)
end

function hash_without_id(chain::RnaChain)::UInt
    return hash((chain.sequence, chain.modifications))
end

function Base.show(io::IO, c::RnaChain)
    print(io, "RnaChain(id=$(c.id), len=$(length(c.sequence)))")
end

function from_dict_rna(d::Dict)::RnaChain
    rna = get(d, "rna", d)
    id  = String(get(rna, "id", "A"))
    seq = String(get(rna, "sequence", ""))
    mods = get(rna, "modifications", Any[])
    modifications = Tuple{String,Int}[
        (String(m["modificationType"]), Int(m["basePosition"])) for m in mods
    ]
    unpaired = _read_msa_field(rna, "unpairedMsa", "unpairedMsaPath")
    desc = get(rna, "description", nothing)
    desc = desc isa String ? desc : nothing
    return RnaChain(id=id, sequence=seq, modifications=modifications,
                    description=desc, unpaired_msa=unpaired)
end

function to_dict(chain::RnaChain)::Dict
    d = Dict{String,Any}(
        "id"       => chain.id,
        "sequence" => chain.sequence,
        "modifications" => [Dict("modificationType"=>c,"basePosition"=>i) for (c,i) in chain.modifications],
    )
    chain.unpaired_msa !== nothing && (d["unpairedMsa"] = chain.unpaired_msa)
    chain.description  !== nothing && (d["description"] = chain.description)
    return Dict("rna" => d)
end

# ──────────────────────────────────────────────────────────────────────────────
# DnaChain
# ──────────────────────────────────────────────────────────────────────────────

struct DnaChain
    id::String
    sequence::String
    modifications::Vector{Tuple{String,Int}}
    description::Union{String,Nothing}
end

function DnaChain(;
    id::String,
    sequence::String,
    modifications::Vector{Tuple{String,Int}} = Tuple{String,Int}[],
    description::Union{String,Nothing} = nothing,
)
    @assert all(c -> c in "ACGTNacgtn", sequence) "DnaChain sequence must be ACGTN"
    @assert all(isuppercase, id) "Chain ID must be uppercase"
    for (ccd, idx) in modifications
        @assert 1 <= idx <= length(sequence) "Modification index $idx out of range"
    end
    return DnaChain(id, uppercase(sequence), modifications, description)
end

function to_ccd_sequence(chain::DnaChain)::Vector{String}
    residues = dna_sequence_to_residues(chain.sequence)
    for (ccd, idx) in chain.modifications
        residues[idx] = ccd
    end
    return residues
end

function fill_missing_fields(chain::DnaChain)::DnaChain
    return chain
end

function hash_without_id(chain::DnaChain)::UInt
    return hash((chain.sequence, chain.modifications))
end

function Base.show(io::IO, c::DnaChain)
    print(io, "DnaChain(id=$(c.id), len=$(length(c.sequence)))")
end

function from_dict_dna(d::Dict)::DnaChain
    dna = get(d, "dna", d)
    id  = String(get(dna, "id", "A"))
    seq = String(get(dna, "sequence", ""))
    mods = get(dna, "modifications", Any[])
    modifications = Tuple{String,Int}[
        (String(m["modificationType"]), Int(m["basePosition"])) for m in mods
    ]
    desc = get(dna, "description", nothing)
    desc = desc isa String ? desc : nothing
    return DnaChain(id=id, sequence=seq, modifications=modifications, description=desc)
end

function to_dict(chain::DnaChain)::Dict
    d = Dict{String,Any}(
        "id"       => chain.id,
        "sequence" => chain.sequence,
        "modifications" => [Dict("modificationType"=>c,"basePosition"=>i) for (c,i) in chain.modifications],
    )
    chain.description !== nothing && (d["description"] = chain.description)
    return Dict("dna" => d)
end

# ──────────────────────────────────────────────────────────────────────────────
# Ligand
# ──────────────────────────────────────────────────────────────────────────────

struct Ligand
    id::String
    ccd_ids::Union{Vector{String},Nothing}
    smiles::Union{String,Nothing}
    description::Union{String,Nothing}
end

function Ligand(;
    id::String,
    ccd_ids::Union{Vector{String},Nothing} = nothing,
    smiles::Union{String,Nothing} = nothing,
    description::Union{String,Nothing} = nothing,
)
    @assert (ccd_ids !== nothing) ⊻ (smiles !== nothing) "Exactly one of ccd_ids or smiles must be set"
    @assert all(isuppercase, id) "Chain ID must be uppercase"
    if ccd_ids !== nothing
        @assert !isempty(ccd_ids) "ccd_ids must be non-empty"
    end
    return Ligand(id, ccd_ids, smiles, description)
end

function hash_without_id(lig::Ligand)::UInt
    return hash((lig.ccd_ids, lig.smiles))
end

function Base.show(io::IO, l::Ligand)
    if l.ccd_ids !== nothing
        print(io, "Ligand(id=$(l.id), ccd=$(join(l.ccd_ids, ',')))")
    else
        print(io, "Ligand(id=$(l.id), smiles=$(l.smiles === nothing ? "?" : l.smiles[1:min(20,length(l.smiles))]))")
    end
end

function from_dict_ligand(d::Dict)::Ligand
    lig = get(d, "ligand", d)
    id  = String(get(lig, "id", "A"))
    ccd = get(lig, "ccdCodes", get(lig, "ccd_ids", nothing))
    smi = get(lig, "smiles", nothing)
    ccd_ids = ccd !== nothing ? String[String(c) for c in ccd] : nothing
    smi_str = smi !== nothing ? String(smi) : nothing
    desc = get(lig, "description", nothing)
    desc = desc isa String ? desc : nothing
    return Ligand(id=id, ccd_ids=ccd_ids, smiles=smi_str, description=desc)
end

function to_dict(lig::Ligand)::Dict
    d = Dict{String,Any}("id" => lig.id)
    lig.ccd_ids !== nothing && (d["ccdCodes"] = lig.ccd_ids)
    lig.smiles  !== nothing && (d["smiles"]   = lig.smiles)
    lig.description !== nothing && (d["description"] = lig.description)
    return Dict("ligand" => d)
end

# ──────────────────────────────────────────────────────────────────────────────
# Atom identifier for bonds
# ──────────────────────────────────────────────────────────────────────────────

"""Atom reference: (chain_id, residue_index, atom_name)"""
const AtomRef = Tuple{String,Int,String}

"""Bond: pair of AtomRef"""
const BondedAtomPair = Tuple{AtomRef, AtomRef}

# ──────────────────────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────────────────────

const AnyChain = Union{ProteinChain,RnaChain,DnaChain,Ligand}

struct Input
    name::String
    chains::Vector{AnyChain}
    rng_seeds::Vector{Int}
    bonded_atom_pairs::Union{Vector{BondedAtomPair},Nothing}
    user_ccd::Union{String,Nothing}
end

_VALID_NAME_CHARS = r"[A-Za-z0-9._\-]"

function Input(;
    name::String,
    chains::Vector{<:AnyChain},
    rng_seeds::Vector{Int},
    bonded_atom_pairs::Union{Vector{BondedAtomPair},Nothing} = nothing,
    user_ccd::Union{String,Nothing} = nothing,
)
    @assert !isempty(name) "Input name must be non-empty"
    @assert occursin(r"[A-Za-z0-9._\-]", name) "Input name must contain at least one valid character"
    @assert !isempty(rng_seeds) "rng_seeds must be non-empty"

    # All chain IDs must be uppercase letters
    for chain in chains
        id = chain.id
        @assert all(c -> isuppercase(c) && isletter(c), id) "Chain ID must be uppercase letter(s), got: $id"
    end

    # No duplicate chain IDs
    ids = [c.id for c in chains]
    @assert length(ids) == length(unique(ids)) "Duplicate chain IDs found: $(join(ids, ','))"

    # Validate user_ccd if provided
    if user_ccd !== nothing
        _validate_user_ccd(user_ccd)
    end

    return Input(name, Vector{AnyChain}(chains), rng_seeds, bonded_atom_pairs, user_ccd)
end

function Base.show(io::IO, inp::Input)
    print(io, "Input(\"$(inp.name)\", $(length(inp.chains)) chains, seeds=$(inp.rng_seeds))")
end

# Chain type accessors
protein_chains(input::Input) = AnyChain[c for c in input.chains if c isa ProteinChain]
rna_chains(input::Input)     = AnyChain[c for c in input.chains if c isa RnaChain]
dna_chains(input::Input)     = AnyChain[c for c in input.chains if c isa DnaChain]
ligands(input::Input)        = AnyChain[c for c in input.chains if c isa Ligand]

"""
    sanitised_name(input::Input) -> String

Replace invalid characters with underscores, truncate to 60 chars.
"""
function sanitised_name(input::Input)::String
    s = replace(input.name, r"[^A-Za-z0-9._\-]" => "_")
    return s[1:min(60, length(s))]
end

"""
    with_multiple_seeds(input::Input, num_seeds::Int) -> Input

Generate `num_seeds` sequential seeds starting from the first seed.
"""
function with_multiple_seeds(input::Input, num_seeds::Int)::Input
    base = isempty(input.rng_seeds) ? 0 : input.rng_seeds[1]
    seeds = collect(base:base+num_seeds-1)
    return Input(name=input.name, chains=input.chains, rng_seeds=seeds,
                 bonded_atom_pairs=input.bonded_atom_pairs, user_ccd=input.user_ccd)
end

# ──────────────────────────────────────────────────────────────────────────────
# JSON serialization
# ──────────────────────────────────────────────────────────────────────────────

"""
    to_json(input::Input) -> String

Serialize Input to JSON string in AlphaFold dialect version 4.
"""
function to_json(input::Input)::String
    sequences = [to_dict(c) for c in input.chains]
    d = Dict{String,Any}(
        "name"        => input.name,
        "sequences"   => sequences,
        "modelSeeds"  => input.rng_seeds,
        "dialect"     => "alphafold3",
        "version"     => 4,
    )
    if input.bonded_atom_pairs !== nothing
        d["bondedAtomPairs"] = [
            [
                [a[1], a[2], a[3]],
                [b[1], b[2], b[3]],
            ]
            for (a, b) in input.bonded_atom_pairs
        ]
    end
    input.user_ccd !== nothing && (d["userCcd"] = input.user_ccd)
    return JSON3.write(d)
end

"""
    from_json(json_str::String) -> Input

Parse Input from JSON string. Supports alphafold3 (v1-4) and alphafoldserver (v1) dialects.
"""
function from_json(json_str::String)::Input
    d = JSON3.read(json_str, Dict{String,Any})
    return _parse_input_dict(d)
end

function _parse_input_dict(d::Dict)::Input
    name = String(get(d, "name", "prediction"))
    seeds_raw = get(d, "modelSeeds", get(d, "seeds", [0]))
    seeds = Int[Int(s) for s in seeds_raw]
    user_ccd = get(d, "userCcd", nothing)
    user_ccd = user_ccd isa String ? user_ccd : nothing

    chains = AnyChain[]
    sequences = get(d, "sequences", get(d, "chains", Any[]))
    for entry in sequences
        entry_dict = Dict{String,Any}(string(k) => v for (k,v) in pairs(entry))
        if haskey(entry_dict, "protein")
            push!(chains, from_dict_protein(entry_dict))
        elseif haskey(entry_dict, "rna")
            push!(chains, from_dict_rna(entry_dict))
        elseif haskey(entry_dict, "dna")
            push!(chains, from_dict_dna(entry_dict))
        elseif haskey(entry_dict, "ligand")
            push!(chains, from_dict_ligand(entry_dict))
        end
    end

    bonds_raw = get(d, "bondedAtomPairs", nothing)
    bonds = if bonds_raw !== nothing
        BondedAtomPair[
            ((String(b[1][1]), Int(b[1][2]), String(b[1][3])),
             (String(b[2][1]), Int(b[2][2]), String(b[2][3])))
            for b in bonds_raw
        ]
    else
        nothing
    end

    return Input(name=name, chains=chains, rng_seeds=seeds,
                 bonded_atom_pairs=bonds, user_ccd=user_ccd)
end

"""
    load_fold_inputs_from_path(path::String) -> Vector{Input}

Load one or more Input objects from a JSON file (supports gzip/xz/zstd).
"""
function load_fold_inputs_from_path(path::String)::Vector{Input}
    text = read_compressed(path)
    # Could be a JSON array or single object
    parsed = JSON3.read(text)
    if parsed isa Vector
        return Input[_parse_input_dict(Dict{String,Any}(string(k)=>v for (k,v) in pairs(item))) for item in parsed]
    else
        d = Dict{String,Any}(string(k) => v for (k,v) in pairs(parsed))
        return [_parse_input_dict(d)]
    end
end

"""
    load_fold_inputs_from_dir(dir::String) -> Vector{Input}

Load all JSON files from a directory.
"""
function load_fold_inputs_from_dir(dir::String)::Vector{Input}
    inputs = Input[]
    for entry in readdir(dir; join=true)
        if endswith(entry, ".json") || endswith(entry, ".json.gz") ||
           endswith(entry, ".json.zst") || endswith(entry, ".json.xz")
            try
                append!(inputs, load_fold_inputs_from_path(entry))
            catch e
                @warn "Failed to load $entry: $e"
            end
        end
    end
    return inputs
end

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

function _read_msa_field(d::Dict, inline_key::String, path_key::String)::Union{String,Nothing}
    inline = get(d, inline_key, nothing)
    inline isa String && !isempty(inline) && return inline
    path = get(d, path_key, nothing)
    path isa String && isfile(path) && return read(path, String)
    return nothing
end

function _dict_to_template(d)::Template
    d_dict = Dict{String,Any}(string(k) => v for (k,v) in pairs(d))
    mmcif = String(get(d_dict, "mmcif", get(d_dict, "mmcifStr", "")))
    # Try loading from path
    if isempty(mmcif)
        p = get(d_dict, "mmcifPath", nothing)
        mmcif = (p isa String && isfile(p)) ? read(p, String) : ""
    end
    map_raw = get(d_dict, "queryToTemplateMap", Dict())
    qtt_map = Dict{Int,Int}(
        Int(parse(Int, string(k))) => Int(v) for (k,v) in pairs(map_raw)
    )
    return Template(mmcif, qtt_map)
end

function _template_to_dict(t::Template)::Dict
    return Dict(
        "mmcif" => t.mmcif,
        "queryToTemplateMap" => Dict(string(k) => v for (k,v) in t.query_to_template_map),
    )
end

function _validate_user_ccd(user_ccd::String)
    required_keys = [
        "_chem_comp.id", "_chem_comp.name", "_chem_comp.type",
        "_chem_comp.formula", "_chem_comp.mon_nstd_parent_comp_id",
        "_chem_comp.pdbx_synonyms", "_chem_comp.formula_weight",
        "_chem_comp_atom.comp_id", "_chem_comp_atom.atom_id",
        "_chem_comp_atom.type_symbol", "_chem_comp_atom.charge",
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
        "_chem_comp_bond.atom_id_1", "_chem_comp_bond.atom_id_2",
        "_chem_comp_bond.value_order", "_chem_comp_bond.pdbx_aromatic_flag",
    ]
    for k in required_keys
        if !occursin(k, user_ccd)
            @warn "user_ccd may be missing required key: $k"
        end
    end
end
