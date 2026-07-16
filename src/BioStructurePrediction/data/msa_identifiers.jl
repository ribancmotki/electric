"""
MSA sequence identifier parsing for paired MSA construction.
"""

"""
    MsaIdentifiers

Parsed accession numbers from an MSA sequence description line.
Used to match sequences across chains for paired MSA construction.
"""
struct MsaIdentifiers
    uniprot_id::Union{String,Nothing}
    uniref_id::Union{String,Nothing}
    mgnify_id::Union{String,Nothing}
    species::Union{String,Nothing}
    raw_description::String
end

"""
    get_identifiers(description::String) -> MsaIdentifiers

Parse a sequence description line and extract accession numbers.

Handles formats:
- UniRef90: ">UniRef90_A0A000|..."
- UniProt: ">sp|P12345|..."
- MGnify: ">MGYP0000000001|..."
- Jackhmmer: ">seq/start-end"
"""
function get_identifiers(description::String)::MsaIdentifiers
    uniprot_id = nothing
    uniref_id  = nothing
    mgnify_id  = nothing
    species    = nothing

    # UniRef90 format: UniRef90_ACCESSION or UniRef90_ACCESSION clustered...
    m = match(r"UniRef90_([A-Z0-9_]+)", description)
    if m !== nothing
        uniref_id = String(m.captures[1])
        # Extract UniProt from UniRef ID if it's a canonical sequence
        m2 = match(r"^([A-Z][0-9][A-Z0-9]{3}[0-9]|[OPQ][0-9][A-Z0-9]{3}[0-9])$", uniref_id)
        m2 !== nothing && (uniprot_id = uniref_id)
    end

    # UniProt Swiss-Prot: sp|ACCESSION|NAME
    m = match(r"(?:sp|tr)\|([A-Z][0-9][A-Z0-9]{3}[0-9])\|(\S+)", description)
    if m !== nothing
        uniprot_id = String(m.captures[1])
        # Extract OS= species if present
        m_os = match(r"OS=([^=]+?)(?:\s+OX=|\s+GN=|\s+PE=|\s*$)", description)
        m_os !== nothing && (species = strip(String(m_os.captures[1])))
    end

    # TrEMBL direct: tr|ACCESSION
    m = match(r"^([A-Z][0-9][A-Z0-9]{3}[0-9])\b", description)
    uniprot_id === nothing && m !== nothing && (uniprot_id = String(m.captures[1]))

    # MGnify: MGYP followed by digits
    m = match(r"(MGYP\d+)", description)
    m !== nothing && (mgnify_id = String(m.captures[1]))

    # Species from OS= annotation
    if species === nothing
        m = match(r"OS=([^=]+?)(?:\s+OX=|\s+GN=|\s+PE=|\s*$)", description)
        m !== nothing && (species = strip(String(m.captures[1])))
    end

    return MsaIdentifiers(uniprot_id, uniref_id, mgnify_id, species, description)
end

"""
    get_pairing_key(ids::MsaIdentifiers) -> Union{String,Nothing}

Return the canonical pairing key for this sequence.
Used to match sequences across chains in paired MSA construction.
Returns UniProt ID if available, then UniRef ID, then MGnify ID, else nothing.
"""
function get_pairing_key(ids::MsaIdentifiers)::Union{String,Nothing}
    ids.uniprot_id !== nothing && return ids.uniprot_id
    ids.uniref_id  !== nothing && return ids.uniref_id
    ids.mgnify_id  !== nothing && return ids.mgnify_id
    return nothing
end
