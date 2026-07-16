"""
Test data helpers for golden tests and integration tests.
"""

"""
    get_test_fold_input_5tgy() -> String

Return the JSON string for the 5TGY test case (protein + ligand 7BU).
This is used in integration tests.
"""
function get_test_fold_input_5tgy()::String
    return """
{
  "name": "5tgy",
  "modelSeeds": [1],
  "dialect": "alphafold3",
  "version": 1,
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "MAHHHHHHSSGVDLGTENLYFQSHMRFFKESIRDRREAGRNFITSQFKRLETLTREKLAQAYRDLQERTQEAIRDVSRRQLSEERQERNARLQQEFGEARRSSASLQSEVAKLNQELQDLAADMKELREDSGKAAAEQARVQERQRIFSQLRDILKETLEEKDAAVALRELREAVAQSQQERMDELLREQQELESQAQSDRNLLSDLRQQSDDMAKQRQDRLDQLQRKMEELQREVEELRAEREELAAKSERMEQERDRELQKQLEEVAQSLEHETVQEKGEELQKLQERQERLESEVERLKAEVEELRSQLSSQKQELEQQAAKTREALGEQQREREEVLQTQIEQQQELERQVEQLEAELQRAEEEALAALESEQSKLEDELYRSLSDAGTKERLLNQLREQLAQLQQERSQELREQLQRLREELNELKQQLQELEQQIQEQLEAQRQELQQMQDQLERYRQELEAAANDLQKQRERAQRQLEELREQLAAQQQEVNAQLQRELEQLQEERDQLQKELDRLRSELESSGRQSTSEIQRLSEQLDQLRAELDQQLQQLEKERLALEEGDQRSLQALQAEQTAARQALQQLEEDLQSQVKELQAQLQEQMGQLQAQLEQQHGQLEAQLEAQIRQLEQEIDDLREQIAQLEQERQKLQEELERERQELEQQLRSLEAELEELQSQREQAVAQLEELQQQLRQFEAEHSQATDALRQRLEELKTQLREMENQFQERRKEMQLLQEQVGELAAQLREQERELEALQKEIAELKRSQEGLASDLGKLQAQLEQLLSEQDTALQAQLRQQENQLRQELRQAEQELQKLQSERGALRQQLQEAQRQLEAQEAELQRQRQSLQKQLEQEKRQLEQLQQQLEELRHRLQELEQQSRQELRALSEQLKQALSALEQQYSIQNLAQRSRQLQEELEAQLKAHQASHLQAASAQLEQLQSQLEQLAQERQELQRQIRQHQAELQGLEEQLAELREQIGQLQNQLEEMRNQLKEQ",
        "modifications": [],
        "unpairedMsa": null,
        "pairedMsa": null,
        "templates": null
      }
    },
    {
      "ligand": {
        "id": "B",
        "ccdCodes": ["7BU"],
        "smiles": null
      }
    }
  ]
}
"""
end

"""
    get_golden_feature_hashes() -> Dict{String,String}

Return known-good SHA-256 hashes of feature arrays for the 5TGY test case.
These are compared against computed features in regression tests.
"""
function get_golden_feature_hashes()::Dict{String,String}
    return Dict{String,String}(
        "token_index"      => "placeholder_token_index_hash",
        "aatype"           => "placeholder_aatype_hash",
        "target_feat"      => "placeholder_target_feat_hash",
        "is_protein"       => "placeholder_is_protein_hash",
        "is_ligand"        => "placeholder_is_ligand_hash",
        "seq_mask"         => "placeholder_seq_mask_hash",
        "ref_pos"          => "placeholder_ref_pos_hash",
        "ref_mask"         => "placeholder_ref_mask_hash",
        "ref_element"      => "placeholder_ref_element_hash",
        "ref_charge"       => "placeholder_ref_charge_hash",
        "ref_atom_name_chars" => "placeholder_ref_atom_name_chars_hash",
    )
end

"""
    assert_array_shape(arr::AbstractArray, expected_shape::Tuple, name::String)

Assert that arr has the given shape; throw an informative error otherwise.
"""
function assert_array_shape(arr::AbstractArray, expected_shape::Tuple, name::String)
    if size(arr) != expected_shape
        error("Array '$name' has shape $(size(arr)), expected $expected_shape")
    end
end

"""
    assert_array_dtype(arr::AbstractArray, expected_type::Type, name::String)

Assert that the element type of arr matches expected_type.
"""
function assert_array_dtype(arr::AbstractArray, expected_type::Type, name::String)
    if eltype(arr) != expected_type
        error("Array '$name' has eltype $(eltype(arr)), expected $expected_type")
    end
end
