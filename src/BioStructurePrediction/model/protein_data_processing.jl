"""
Protein-specific data processing for featurisation.
"""

"""
    process_protein_chain_features(
        chain::ProteinChain,
        ccd::Ccd,
        msa_config::MsaConfig,
    ) -> Dict{String,Any}

Process a single protein chain into intermediate feature data.
Returns a dict with:
  - residue_types: Vector{String}
  - msa: Msa (unpaired)
  - paired_msa: Msa
  - templates: Vector{TemplateHitInput}
  - ref_pos: (n_res, 37, 3) Float32
  - ref_mask: (n_res, 37) Bool
  - ref_element: (n_res, 37) Int32
  - ref_charge: (n_res, 37) Float32
"""
function process_protein_chain_features(
    chain::ProteinChain,
    ccd::Ccd,
    msa_config::MsaConfig,
)::Dict{String,Any}
    residue_types = protein_sequence_to_residues(chain.sequence)

    # Parse MSA
    unpaired_msa = if chain.unpaired_msa !== nothing && !isempty(chain.unpaired_msa)
        Msa_from_a3m(chain.unpaired_msa)
    else
        query_aln = uppercase(chain.sequence)
        Msa([query_aln], ["query"], zeros(Int, 1, length(query_aln)))
    end

    paired_msa = if chain.paired_msa !== nothing && !isempty(chain.paired_msa)
        Msa_from_a3m(chain.paired_msa)
    else
        Msa(String[], String[], zeros(Int, 0, 0))
    end

    # Truncate MSA
    unpaired_msa = truncate_msa(unpaired_msa, msa_config.max_unpaired_sequences)
    paired_msa   = truncate_msa(paired_msa,   msa_config.max_paired_sequences)

    # Reference atom positions from CCD
    n_res     = length(residue_types)
    ref_pos   = zeros(Float32, n_res, NUM_ATOM_SLOTS, 3)
    ref_mask  = falses(n_res, NUM_ATOM_SLOTS)
    ref_elem  = zeros(Int32, n_res, NUM_ATOM_SLOTS)
    ref_charge = zeros(Float32, n_res, NUM_ATOM_SLOTS)

    for (i, res) in enumerate(residue_types)
        comp = get_component(ccd, res)
        comp === nothing && continue
        atom_order_for_res = get(ATOM_ORDER, res, String[])
        atom_name_to_ccd = Dict(a.atom_id => a for a in comp.atoms)
        for (j, atom_name) in enumerate(atom_order_for_res)
            j > NUM_ATOM_SLOTS && break
            isempty(atom_name) && continue
            a = get(atom_name_to_ccd, atom_name, nothing)
            a === nothing && continue
            ref_pos[i, j, 1] = a.ideal_x
            ref_pos[i, j, 2] = a.ideal_y
            ref_pos[i, j, 3] = a.ideal_z
            ref_mask[i, j]   = true
            ref_elem[i, j]   = Int32(get_element_index(a.element))
            ref_charge[i, j] = a.charge
        end
    end

    templates = chain.templates !== nothing ? chain.templates : TemplateHitInput[]

    return Dict{String,Any}(
        "residue_types"  => residue_types,
        "unpaired_msa"   => unpaired_msa,
        "paired_msa"     => paired_msa,
        "templates"      => templates,
        "ref_pos"        => ref_pos,
        "ref_mask"       => ref_mask,
        "ref_element"    => ref_elem,
        "ref_charge"     => ref_charge,
    )
end

"""
    process_rna_chain_features(chain::RnaChain, ccd::Ccd, msa_config::MsaConfig) -> Dict{String,Any}

Process a single RNA chain into intermediate feature data.
"""
function process_rna_chain_features(
    chain::RnaChain,
    ccd::Ccd,
    msa_config::MsaConfig,
)::Dict{String,Any}
    residue_types = rna_sequence_to_residues(chain.sequence)
    n_res = length(residue_types)

    unpaired_msa = if chain.unpaired_msa !== nothing && !isempty(chain.unpaired_msa)
        Msa_from_a3m(chain.unpaired_msa)
    else
        query_aln = uppercase(chain.sequence)
        Msa([query_aln], ["query"], zeros(Int, 1, length(query_aln)))
    end
    unpaired_msa = truncate_msa(unpaired_msa, msa_config.max_unpaired_sequences)

    ref_pos    = zeros(Float32, n_res, NUM_ATOM_SLOTS, 3)
    ref_mask   = falses(n_res, NUM_ATOM_SLOTS)
    ref_elem   = zeros(Int32, n_res, NUM_ATOM_SLOTS)
    ref_charge = zeros(Float32, n_res, NUM_ATOM_SLOTS)

    for (i, res) in enumerate(residue_types)
        comp = get_component(ccd, res)
        comp === nothing && continue
        atom_order_for_res = get(ATOM_ORDER, res, String[])
        atom_name_to_ccd = Dict(a.atom_id => a for a in comp.atoms)
        for (j, atom_name) in enumerate(atom_order_for_res)
            j > NUM_ATOM_SLOTS && break
            isempty(atom_name) && continue
            a = get(atom_name_to_ccd, atom_name, nothing)
            a === nothing && continue
            ref_pos[i, j, 1] = a.ideal_x
            ref_pos[i, j, 2] = a.ideal_y
            ref_pos[i, j, 3] = a.ideal_z
            ref_mask[i, j]   = true
            ref_elem[i, j]   = Int32(get_element_index(a.element))
            ref_charge[i, j] = a.charge
        end
    end

    return Dict{String,Any}(
        "residue_types" => residue_types,
        "unpaired_msa"  => unpaired_msa,
        "paired_msa"    => Msa(String[], String[], zeros(Int, 0, 0)),
        "templates"     => TemplateHitInput[],
        "ref_pos"       => ref_pos,
        "ref_mask"      => ref_mask,
        "ref_element"   => ref_elem,
        "ref_charge"    => ref_charge,
    )
end

"""
    process_dna_chain_features(chain::DnaChain, ccd::Ccd) -> Dict{String,Any}

Process a single DNA chain into intermediate feature data.
"""
function process_dna_chain_features(chain::DnaChain, ccd::Ccd)::Dict{String,Any}
    residue_types = dna_sequence_to_residues(chain.sequence)
    n_res = length(residue_types)

    ref_pos    = zeros(Float32, n_res, NUM_ATOM_SLOTS, 3)
    ref_mask   = falses(n_res, NUM_ATOM_SLOTS)
    ref_elem   = zeros(Int32, n_res, NUM_ATOM_SLOTS)
    ref_charge = zeros(Float32, n_res, NUM_ATOM_SLOTS)

    for (i, res) in enumerate(residue_types)
        comp = get_component(ccd, res)
        comp === nothing && continue
        atom_order_for_res = get(ATOM_ORDER, res, String[])
        atom_name_to_ccd = Dict(a.atom_id => a for a in comp.atoms)
        for (j, atom_name) in enumerate(atom_order_for_res)
            j > NUM_ATOM_SLOTS && break
            isempty(atom_name) && continue
            a = get(atom_name_to_ccd, atom_name, nothing)
            a === nothing && continue
            ref_pos[i, j, 1] = a.ideal_x
            ref_pos[i, j, 2] = a.ideal_y
            ref_pos[i, j, 3] = a.ideal_z
            ref_mask[i, j]   = true
            ref_elem[i, j]   = Int32(get_element_index(a.element))
            ref_charge[i, j] = a.charge
        end
    end

    return Dict{String,Any}(
        "residue_types" => residue_types,
        "ref_pos"       => ref_pos,
        "ref_mask"      => ref_mask,
        "ref_element"   => ref_elem,
        "ref_charge"    => ref_charge,
    )
end

"""
    process_ligand_features(
        lig::LigandEntity,
        ccd::Ccd;
        conformer_max_iterations::Union{Int,Nothing} = nothing,
        ref_max_modified_date::Union{Date,Nothing} = nothing
    ) -> Dict{String,Any}

Process a single ligand into intermediate feature data.
Generates or retrieves 3D conformer coordinates.
"""
function process_ligand_features(
    lig::LigandEntity,
    ccd::Ccd;
    conformer_max_iterations::Union{Int,Nothing} = nothing,
    ref_max_modified_date::Union{Date,Nothing}   = nothing,
)::Dict{String,Any}
    # Determine atom data
    if !isempty(lig.ccd_codes)
        # CCD ligand
        comp_id = first(lig.ccd_codes)
        comp_data = get_component_atoms(ccd, comp_id)
        atom_names = comp_data.atom_names
        elements   = comp_data.elements
        charges    = comp_data.charges
        n_atoms    = length(atom_names)

        # Try ideal coordinates first
        coords = comp_data.ideal_pos  # (n_atoms, 3)

        # Check if all zero (no ideal coords)
        if all(coords .== 0f0)
            coords = comp_data.model_pos
        end

        ref_pos   = zeros(Float32, n_atoms, NUM_ATOM_SLOTS, 3)
        ref_mask  = falses(n_atoms, NUM_ATOM_SLOTS)
        ref_elem  = zeros(Int32, n_atoms, NUM_ATOM_SLOTS)
        ref_charge = zeros(Float32, n_atoms, NUM_ATOM_SLOTS)

        for i in 1:n_atoms
            ref_pos[i, 1, 1]   = coords[i, 1]
            ref_pos[i, 1, 2]   = coords[i, 2]
            ref_pos[i, 1, 3]   = coords[i, 3]
            ref_mask[i, 1]     = true
            ref_elem[i, 1]     = Int32(get_element_index(elements[i]))
            ref_charge[i, 1]   = charges[i]
        end

        # One token per heavy atom
        ligand_residue_types = fill(comp_id, n_atoms)

    elseif lig.smiles !== nothing
        # SMILES ligand — generate simple atom-level features
        # In a full implementation this would use RDKit; here we parse the SMILES
        atom_names, elements, n_atoms = parse_smiles_atoms(lig.smiles)

        ref_pos    = zeros(Float32, n_atoms, NUM_ATOM_SLOTS, 3)
        ref_mask   = falses(n_atoms, NUM_ATOM_SLOTS)
        ref_elem   = zeros(Int32, n_atoms, NUM_ATOM_SLOTS)
        ref_charge = zeros(Float32, n_atoms, NUM_ATOM_SLOTS)

        for i in 1:n_atoms
            ref_mask[i, 1]   = true
            ref_elem[i, 1]   = Int32(get_element_index(elements[i]))
        end

        ligand_residue_types = fill("UNK", n_atoms)
    else
        error("Ligand has neither ccdCodes nor smiles: $(lig.ids)")
    end

    return Dict{String,Any}(
        "residue_types" => ligand_residue_types,
        "ref_pos"       => ref_pos,
        "ref_mask"      => ref_mask,
        "ref_element"   => ref_elem,
        "ref_charge"    => ref_charge,
        "is_ligand"     => true,
    )
end

"""
    parse_smiles_atoms(smiles::String) -> Tuple{Vector{String},Vector{String},Int}

Parse heavy atom names and elements from a SMILES string.
Returns (atom_names, elements, n_atoms).
This is a minimal parser; a full implementation uses RDKit.
"""
function parse_smiles_atoms(smiles::String)::Tuple{Vector{String},Vector{String},Int}
    elements   = String[]
    atom_names = String[]

    i = 1
    while i <= length(smiles)
        c = smiles[i]
        # Skip non-atom characters
        if c ∈ ('+', '-', '(', ')', '[', ']', '=', '#', ':', '.', '/', '\\', '@', '%')
            i += 1
            continue
        end
        if isdigit(c)
            i += 1
            continue
        end
        # Atom symbol
        if isuppercase(c)
            # Try two-character element
            if i < length(smiles) && islowercase(smiles[i+1])
                elem = string(c, smiles[i+1])
                if haskey(PERIODIC_TABLE, elem)
                    push!(elements, elem)
                    push!(atom_names, "$(elem)$(length(elements))")
                    i += 2
                    continue
                end
            end
            # Single character element
            elem = string(c)
            if haskey(PERIODIC_TABLE, elem) && elem != "H"
                push!(elements, elem)
                push!(atom_names, "$(elem)$(length(elements))")
            end
            i += 1
        else
            i += 1
        end
    end

    n_atoms = length(elements)
    return atom_names, elements, n_atoms
end
