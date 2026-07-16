"""
protein_data_processing.jl — Per-chain feature processors.
"""

using Logging

# ──────────────────────────────────────────────────────────────────────────────
# Protein chain processing
# ──────────────────────────────────────────────────────────────────────────────

"""
    process_protein_chain(chain::ProteinChain, ccd::Ccd;
                           max_msa_seqs=512, max_template_hits=4) -> Dict{String,Any}

Process a protein chain into model-ready features.
"""
function process_protein_chain(
    chain::ProteinChain,
    ccd::Ccd;
    max_msa_seqs::Int       = 512,
    max_template_hits::Int  = 4,
    max_extra_msa_seqs::Int = 1024,
)::Dict{String,Any}
    seq = chain.sequence
    n   = length(seq)

    # ── MSA ──────────────────────────────────────────────────────────────────
    msa_a3m = chain.unpaired_msa !== nothing ? chain.unpaired_msa : ">q\n$seq\n"
    msa_obj = msa_from_a3m(seq, PROTEIN_CHAIN, msa_a3m)

    # Paired MSA
    paired_a3m = chain.paired_msa !== nothing ? chain.paired_msa : ">q\n$seq\n"
    paired_msa_obj = msa_from_a3m(seq, PROTEIN_CHAIN, paired_a3m)

    # Extra MSA (sequences beyond main MSA)
    msa_full   = truncate_msa(msa_obj, max_msa_seqs + max_extra_msa_seqs)
    main_msa   = truncate_msa(msa_full, max_msa_seqs)
    extra_msa  = Msa(
        msa_full.sequences[min(max_msa_seqs+1, end):end],
        msa_full.descriptions[min(max_msa_seqs+1, end):end],
        msa_full.deletion_matrix[min(max_msa_seqs+1, end):end, :],
        msa_full.query_sequence,
        msa_full.chain_poly_type,
    )

    msa_feat       = make_msa_features(main_msa)
    extra_msa_feat = make_extra_msa_features(extra_msa; max_extra_msa=max_extra_msa_seqs)

    # ── Sequence features ────────────────────────────────────────────────────
    ccd_seq = protein_sequence_to_residues(seq)
    seq_onehot = zeros(Float32, n, POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP)
    for (i, rn) in enumerate(ccd_seq)
        idx = get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, rn, 0)
        idx > 0 && (seq_onehot[i, idx] = 1f0)
    end

    # ── Post-translational modifications ────────────────────────────────────
    ptm_mask = zeros(Float32, n)
    for ptm in chain.ptms
        if ptm.position <= n
            ptm_mask[ptm.position] = 1f0
        end
    end

    return Dict{String,Any}(
        "sequence_one_hot"        => seq_onehot,
        "ptm_mask"                => ptm_mask,
        "chain_id"                => chain.id,
        "chain_poly_type"         => PROTEIN_CHAIN,
        "unpaired_msa"            => main_msa,
        "paired_msa"              => paired_msa_obj,
        "extra_msa"               => extra_msa,
        "msa_features"            => msa_feat,
        "extra_msa_features"      => extra_msa_feat,
    )
end

"""
    process_rna_chain(chain::RnaChain, ccd::Ccd;
                       max_msa_seqs=512) -> Dict{String,Any}
"""
function process_rna_chain(
    chain::RnaChain,
    ccd::Ccd;
    max_msa_seqs::Int = 512,
)::Dict{String,Any}
    seq = chain.sequence
    n   = length(seq)

    msa_a3m = chain.unpaired_msa !== nothing ? chain.unpaired_msa : ">q\n$seq\n"
    msa_obj = msa_from_a3m(seq, RNA_CHAIN, msa_a3m)
    msa_obj = truncate_msa(msa_obj, max_msa_seqs)
    msa_feat = make_msa_features(msa_obj)

    ccd_seq = rna_sequence_to_residues(seq)
    seq_onehot = zeros(Float32, n, POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP)
    for (i, rn) in enumerate(ccd_seq)
        idx = get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, rn, 0)
        idx > 0 && (seq_onehot[i, idx] = 1f0)
    end

    return Dict{String,Any}(
        "sequence_one_hot" => seq_onehot,
        "chain_id"         => chain.id,
        "chain_poly_type"  => RNA_CHAIN,
        "unpaired_msa"     => msa_obj,
        "msa_features"     => msa_feat,
    )
end

"""
    process_dna_chain(chain::DnaChain, ccd::Ccd) -> Dict{String,Any}
"""
function process_dna_chain(chain::DnaChain, ccd::Ccd)::Dict{String,Any}
    seq = chain.sequence
    n   = length(seq)

    ccd_seq = dna_sequence_to_residues(seq)
    seq_onehot = zeros(Float32, n, POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP)
    for (i, rn) in enumerate(ccd_seq)
        idx = get(POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP, rn, 0)
        idx > 0 && (seq_onehot[i, idx] = 1f0)
    end

    return Dict{String,Any}(
        "sequence_one_hot" => seq_onehot,
        "chain_id"         => chain.id,
        "chain_poly_type"  => DNA_CHAIN,
    )
end

"""
    process_ligand(ligand::Ligand, ccd::Ccd) -> Dict{String,Any}
"""
function process_ligand(ligand::Ligand, ccd::Ccd)::Dict{String,Any}
    comp = get(ccd, ligand.ccd_id, nothing)
    smiles = ligand.smiles

    n_atoms = 0
    atom_names = String[]
    atom_elements = String[]

    if comp !== nothing
        for a in comp.atoms
            a.type_symbol in ("H", "D") && continue
            push!(atom_names, a.atom_id)
            push!(atom_elements, a.type_symbol)
            n_atoms += 1
        end
    elseif smiles !== nothing
        atoms = smiles_to_atoms(smiles; include_hydrogens=false)
        for a in atoms
            push!(atom_names, a["atom_id"])
            push!(atom_elements, a["type_symbol"])
            n_atoms += 1
        end
    end

    seq_onehot = zeros(Float32, n_atoms, POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP)
    # Ligand atoms: use element-based one-hot (no standard residue type)

    return Dict{String,Any}(
        "sequence_one_hot"  => seq_onehot,
        "chain_id"          => ligand.id,
        "chain_poly_type"   => LIGAND_CHAIN,
        "ccd_id"            => ligand.ccd_id,
        "smiles"            => smiles,
        "atom_names"        => atom_names,
        "atom_elements"     => atom_elements,
        "n_atoms"           => n_atoms,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# PTM (post-translational modification) utilities
# ──────────────────────────────────────────────────────────────────────────────

"""
    apply_ptms(residues::Vector{String}, ptms::Vector{ProteinModification}) -> Vector{String}

Apply PTMs to residue sequence by replacing residue names.
"""
function apply_ptms(residues::Vector{String},
                    ptms::Vector{ProteinModification})::Vector{String}
    result = copy(residues)
    for ptm in ptms
        if 1 <= ptm.position <= length(result)
            result[ptm.position] = ptm.ptm_type
        end
    end
    return result
end
