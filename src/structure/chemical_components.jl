"""
chemical_components.jl — Structure-layer chemical component utilities.
"""

"""
    get_all_atoms_in_entry(ccd, res_name::String) -> Dict{String,Vector{String}}

Return atom data for a CCD entry. Result has keys:
- "_chem_comp_atom.atom_id": Vector{String}
- "_chem_comp_atom.type_symbol": Vector{String}
Returns empty vectors if the component is not found.
"""
function get_all_atoms_in_entry(ccd, res_name::String)::Dict{String,Vector{String}}
    comp = get(ccd, res_name, nothing)
    comp === nothing && return Dict{String,Vector{String}}(
        "_chem_comp_atom.atom_id"    => String[],
        "_chem_comp_atom.type_symbol" => String[],
        "_chem_comp_atom.charge"      => String[],
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal" => String[],
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal" => String[],
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal" => String[],
    )
    atoms = comp.atoms
    return Dict{String,Vector{String}}(
        "_chem_comp_atom.atom_id"    => [a.atom_id for a in atoms],
        "_chem_comp_atom.type_symbol" => [a.type_symbol for a in atoms],
        "_chem_comp_atom.charge"      => [string(a.charge) for a in atoms],
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal" => [string(a.x_ideal) for a in atoms],
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal" => [string(a.y_ideal) for a in atoms],
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal" => [string(a.z_ideal) for a in atoms],
    )
end

"""
    get_residue_component_atoms(ccd, res_name::String; include_hydrogens=false)
        -> Vector{NamedTuple}

Return list of atom descriptors for a residue component.
"""
function get_residue_component_atoms(ccd, res_name::String;
                                     include_hydrogens::Bool=false)
    comp = get(ccd, res_name, nothing)
    comp === nothing && return NamedTuple[]
    atoms = comp.atoms
    if !include_hydrogens
        atoms = filter(a -> a.type_symbol != "H" && a.type_symbol != "D", atoms)
    end
    return [(
        atom_id = a.atom_id,
        type_symbol = a.type_symbol,
        charge = a.charge,
        x_ideal = a.x_ideal,
        y_ideal = a.y_ideal,
        z_ideal = a.z_ideal,
        leaving_atom = a.leaving_atom_flag,
    ) for a in atoms]
end

"""
    compare_chirality(res_name::String, res_atoms::Dict{String,Array{Float32,1}}, ccd) -> Bool

Check whether the chirality of the observed residue matches the CCD template.
Uses cross products of bond vectors at chiral centers.
"""
function compare_chirality(res_name::String,
                           obs_coords::Dict{String,Vector{Float32}},
                           ccd)::Bool
    centers = get_chiral_centers(res_name)
    isempty(centers) && return true

    ideal_positions = get_ideal_positions(ccd, res_name)
    for (center, n1, n2, n3) in centers
        # Check both observed and ideal have the required atoms
        all(k -> haskey(obs_coords, k), (center, n1, n2, n3)) || continue
        all(k -> haskey(ideal_positions, k), (center, n1, n2, n3)) || continue

        obs_sign  = _chirality_sign(obs_coords, center, n1, n2, n3)
        ideal_sign = _chirality_sign_tuple(ideal_positions, center, n1, n2, n3)

        obs_sign != 0 && ideal_sign != 0 && obs_sign != ideal_sign && return false
    end
    return true
end

function _chirality_sign(coords::Dict{String,Vector{Float32}},
                         center::String, n1::String, n2::String, n3::String)::Int
    c  = coords[center]
    a1 = coords[n1] .- c
    a2 = coords[n2] .- c
    a3 = coords[n3] .- c
    cross12 = [
        a1[2]*a2[3] - a1[3]*a2[2],
        a1[3]*a2[1] - a1[1]*a2[3],
        a1[1]*a2[2] - a1[2]*a2[1],
    ]
    det = cross12[1]*a3[1] + cross12[2]*a3[2] + cross12[3]*a3[3]
    det > 1e-6 && return 1
    det < -1e-6 && return -1
    return 0
end

function _chirality_sign_tuple(coords::Dict{String,NTuple{3,Float32}},
                               center::String, n1::String, n2::String, n3::String)::Int
    c  = collect(coords[center])
    c1 = collect(coords[n1]) .- c
    c2 = collect(coords[n2]) .- c
    c3 = collect(coords[n3]) .- c
    cross12 = [
        c1[2]*c2[3] - c1[3]*c2[2],
        c1[3]*c2[1] - c1[1]*c2[3],
        c1[1]*c2[2] - c1[2]*c2[1],
    ]
    det = cross12[1]*c3[1] + cross12[2]*c3[2] + cross12[3]*c3[3]
    det > 1e-6 && return 1
    det < -1e-6 && return -1
    return 0
end

"""
    compute_chirality_sign(pos_center, pos_n1, pos_n2, pos_n3) -> Int

Compute the chirality sign (+1 or -1) from four 3D positions.
Returns 0 if degenerate.
"""
function compute_chirality_sign(pc::AbstractVector{<:Real},
                                 pn1::AbstractVector{<:Real},
                                 pn2::AbstractVector{<:Real},
                                 pn3::AbstractVector{<:Real})::Int
    a1 = pn1 .- pc
    a2 = pn2 .- pc
    a3 = pn3 .- pc
    cross = [
        a1[2]*a2[3] - a1[3]*a2[2],
        a1[3]*a2[1] - a1[1]*a2[3],
        a1[1]*a2[2] - a1[2]*a2[1],
    ]
    det = cross[1]*a3[1] + cross[2]*a3[2] + cross[3]*a3[3]
    det > 1e-6 && return 1
    det < -1e-6 && return -1
    return 0
end
