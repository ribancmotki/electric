"""
msa_identifiers.jl — MSA sequence description parsing for pairing.
"""

"""
    MsaIdentifiers

Identifiers extracted from MSA sequence descriptions for inter-chain pairing.
"""
struct MsaIdentifiers
    sequence_id::String
    species_id::String
    uniprot_id::String
end

function Base.show(io::IO, m::MsaIdentifiers)
    print(io, "MsaIdentifiers(seq=$(m.sequence_id), species=$(m.species_id), uniprot=$(m.uniprot_id))")
end

"""
    get_identifiers(description::String) -> MsaIdentifiers

Parse a UniProt/UniRef MSA sequence description to extract identifiers.
Expected format: "UniRef100_XXXXX/start-end TaxID=NNNN ..."
"""
function get_identifiers(description::String)::MsaIdentifiers
    seq_id = description
    species_id = ""
    uniprot_id = ""

    # Extract sequence ID (first token before whitespace)
    parts = split(description, r"\s+")
    seq_id = isempty(parts) ? description : parts[1]

    # Extract species/TaxID
    tax_m = match(r"TaxID=(\d+)", description)
    if tax_m !== nothing
        species_id = tax_m.captures[1]
    end

    # Extract OX= field (alternative TaxID format)
    ox_m = match(r"\bOX=(\d+)\b", description)
    if ox_m !== nothing
        species_id = ox_m.captures[1]
    end

    # Extract UniProt accession from sequence ID
    up_m = match(r"[_|]([A-Z][0-9][A-Z0-9]{3}[0-9]|[OPQ][0-9][A-Z0-9]{3}[0-9])[_|/]?", seq_id)
    if up_m !== nothing
        uniprot_id = up_m.captures[1]
    end

    return MsaIdentifiers(String(seq_id), String(species_id), String(uniprot_id))
end

"""
    get_pairing_key(identifiers::MsaIdentifiers) -> String

Return the key used for inter-chain MSA pairing.
Prefer species_id (TaxID), fall back to uniprot_id.
"""
function get_pairing_key(identifiers::MsaIdentifiers)::String
    !isempty(identifiers.species_id) && return identifiers.species_id
    !isempty(identifiers.uniprot_id) && return identifiers.uniprot_id
    return identifiers.sequence_id
end

"""
    parse_species_from_msa(msa::Msa) -> Vector{String}

Extract species identifiers from all sequences in an MSA.
"""
function parse_species_from_msa(msa::Msa)::Vector{String}
    return [get_pairing_key(get_identifiers(d)) for d in msa.descriptions]
end
