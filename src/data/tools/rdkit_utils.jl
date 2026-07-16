"""
rdkit_utils.jl — RDKit molecule utilities via Python interop (PyCall.jl) or fallback.
"""

using Logging

# Attempt to load PyCall and RDKit; fall back gracefully if unavailable.
const _RDKIT_AVAILABLE = Ref{Bool}(false)
const _Chem = Ref{Any}(nothing)
const _AllChem = Ref{Any}(nothing)

function _try_load_rdkit()
    if _RDKIT_AVAILABLE[]
        return true
    end
    try
        @eval using PyCall
        rdkit = PyCall.pyimport_conda("rdkit.Chem", "rdkit")
        _Chem[] = rdkit
        allchem = PyCall.pyimport_conda("rdkit.Chem.AllChem", "rdkit")
        _AllChem[] = allchem
        _RDKIT_AVAILABLE[] = true
        return true
    catch e
        @debug "RDKit not available via PyCall: $e"
        return false
    end
end

"""
    smiles_to_atoms(smiles::String; include_hydrogens=false) -> Vector{Dict{String,Any}}

Parse a SMILES string and return atom data.
Each atom: Dict("atom_id"=>"C1", "type_symbol"=>"C", "charge"=>0.0,
               "x_ideal"=>0.0, "y_ideal"=>0.0, "z_ideal"=>0.0)
Uses RDKit for 3D conformer generation.
Falls back to topology-only parsing if RDKit is unavailable.
"""
function smiles_to_atoms(smiles::String;
                          include_hydrogens::Bool=false,
                          max_iterations::Int=2000)::Vector{Dict{String,Any}}
    if _try_load_rdkit()
        return _smiles_to_atoms_rdkit(smiles; include_hydrogens, max_iterations)
    else
        return _smiles_to_atoms_fallback(smiles; include_hydrogens)
    end
end

function _smiles_to_atoms_rdkit(smiles::String;
                                 include_hydrogens::Bool=false,
                                 max_iterations::Int=2000)::Vector{Dict{String,Any}}
    Chem = _Chem[]
    AllChem = _AllChem[]
    try
        mol = Chem.MolFromSmiles(smiles)
        mol === nothing && return _smiles_to_atoms_fallback(smiles; include_hydrogens)

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iterations)

        if !include_hydrogens
            mol = Chem.RemoveHs(mol)
        end

        conf = mol.GetConformer()
        atoms = Dict{String,Any}[]
        for atom in mol.GetAtoms()
            idx  = atom.GetIdx()
            pos  = conf.GetAtomPosition(idx)
            symbol = atom.GetSymbol()
            charge = Float32(atom.GetFormalCharge())
            push!(atoms, Dict{String,Any}(
                "atom_id"    => "$symbol$(idx+1)",
                "type_symbol"=> symbol,
                "charge"     => charge,
                "x_ideal"    => Float32(pos.x),
                "y_ideal"    => Float32(pos.y),
                "z_ideal"    => Float32(pos.z),
            ))
        end
        return atoms
    catch e
        @warn "RDKit SMILES processing failed: $e"
        return _smiles_to_atoms_fallback(smiles; include_hydrogens)
    end
end

"""
Fallback atom parser: extract atoms from SMILES without 3D coordinates.
Returns atoms with zero coordinates.
"""
function _smiles_to_atoms_fallback(smiles::String;
                                    include_hydrogens::Bool=false)::Vector{Dict{String,Any}}
    # Scan SMILES for element symbols
    atoms = Dict{String,Any}[]
    i = 1
    idx = 0
    while i <= length(smiles)
        c = smiles[i]
        if isupper(c)
            # Possible element symbol
            symbol = string(c)
            if i < length(smiles) && islower(smiles[i+1])
                symbol = string(c, smiles[i+1])
                i += 1
            end
            if symbol == "H" && !include_hydrogens
                i += 1
                continue
            end
            idx += 1
            push!(atoms, Dict{String,Any}(
                "atom_id"     => "$symbol$idx",
                "type_symbol" => symbol,
                "charge"      => 0f0,
                "x_ideal"     => 0f0,
                "y_ideal"     => 0f0,
                "z_ideal"     => 0f0,
            ))
        end
        i += 1
    end
    return atoms
end

"""
    validate_smiles(smiles::String) -> Bool

Return true if the SMILES is parseable.
"""
function validate_smiles(smiles::String)::Bool
    isempty(smiles) && return false
    _try_load_rdkit() || return !isempty(_smiles_to_atoms_fallback(smiles))
    try
        Chem = _Chem[]
        mol = Chem.MolFromSmiles(smiles)
        return mol !== nothing
    catch
        return false
    end
end
