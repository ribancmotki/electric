"""
MSA sampling configuration.
"""

"""
    MsaConfig

Controls how MSA sequences are sampled and truncated during featurisation.

Fields:
- max_paired_sequences: maximum number of paired MSA rows (for multi-chain jobs)
- max_unpaired_sequences: maximum number of unpaired MSA rows per chain
- max_extra_msa_sequences: maximum number of extra (lower-quality) MSA rows
"""
struct MsaConfig
    max_paired_sequences::Int
    max_unpaired_sequences::Int
    max_extra_msa_sequences::Int
end

"""
    default_msa_config() -> MsaConfig

Return the default MSA configuration.
"""
function default_msa_config()::MsaConfig
    return MsaConfig(
        2048,   # max_paired_sequences
        2048,   # max_unpaired_sequences
        16384,  # max_extra_msa_sequences
    )
end
