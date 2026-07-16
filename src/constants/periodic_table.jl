"""
periodic_table.jl — element symbol to atomic number mapping for all 118 elements.
"""

const ELEMENT_TO_ATOMIC_NUMBER = Dict{String,Int}(
    "H"  => 1,   "He" => 2,   "Li" => 3,   "Be" => 4,   "B"  => 5,
    "C"  => 6,   "N"  => 7,   "O"  => 8,   "F"  => 9,   "Ne" => 10,
    "Na" => 11,  "Mg" => 12,  "Al" => 13,  "Si" => 14,  "P"  => 15,
    "S"  => 16,  "Cl" => 17,  "Ar" => 18,  "K"  => 19,  "Ca" => 20,
    "Sc" => 21,  "Ti" => 22,  "V"  => 23,  "Cr" => 24,  "Mn" => 25,
    "Fe" => 26,  "Co" => 27,  "Ni" => 28,  "Cu" => 29,  "Zn" => 30,
    "Ga" => 31,  "Ge" => 32,  "As" => 33,  "Se" => 34,  "Br" => 35,
    "Kr" => 36,  "Rb" => 37,  "Sr" => 38,  "Y"  => 39,  "Zr" => 40,
    "Nb" => 41,  "Mo" => 42,  "Tc" => 43,  "Ru" => 44,  "Rh" => 45,
    "Pd" => 46,  "Ag" => 47,  "Cd" => 48,  "In" => 49,  "Sn" => 50,
    "Sb" => 51,  "Te" => 52,  "I"  => 53,  "Xe" => 54,  "Cs" => 55,
    "Ba" => 56,  "La" => 57,  "Ce" => 58,  "Pr" => 59,  "Nd" => 60,
    "Pm" => 61,  "Sm" => 62,  "Eu" => 63,  "Gd" => 64,  "Tb" => 65,
    "Dy" => 66,  "Ho" => 67,  "Er" => 68,  "Tm" => 69,  "Yb" => 70,
    "Lu" => 71,  "Hf" => 72,  "Ta" => 73,  "W"  => 74,  "Re" => 75,
    "Os" => 76,  "Ir" => 77,  "Pt" => 78,  "Au" => 79,  "Hg" => 80,
    "Tl" => 81,  "Pb" => 82,  "Bi" => 83,  "Po" => 84,  "At" => 85,
    "Rn" => 86,  "Fr" => 87,  "Ra" => 88,  "Ac" => 89,  "Th" => 90,
    "Pa" => 91,  "U"  => 92,  "Np" => 93,  "Pu" => 94,  "Am" => 95,
    "Cm" => 96,  "Bk" => 97,  "Cf" => 98,  "Es" => 99,  "Fm" => 100,
    "Md" => 101, "No" => 102, "Lr" => 103, "Rf" => 104, "Db" => 105,
    "Sg" => 106, "Bh" => 107, "Hs" => 108, "Mt" => 109, "Ds" => 110,
    "Rg" => 111, "Cn" => 112, "Nh" => 113, "Fl" => 114, "Mc" => 115,
    "Lv" => 116, "Ts" => 117, "Og" => 118,
)

# Reverse mapping: atomic number → element symbol
const ATOMIC_NUMBER_TO_ELEMENT = Dict{Int,String}(
    v => k for (k, v) in ELEMENT_TO_ATOMIC_NUMBER
)

"""
    get_atomic_number(element::String) -> Int

Return the atomic number for an element symbol. Returns 0 for unknown elements.
"""
function get_atomic_number(element::String)::Int
    return get(ELEMENT_TO_ATOMIC_NUMBER, element, 0)
end

"""
    get_element_symbol(atomic_number::Int) -> String

Return the element symbol for an atomic number. Returns "" for unknown numbers.
"""
function get_element_symbol(atomic_number::Int)::String
    return get(ATOMIC_NUMBER_TO_ELEMENT, atomic_number, "")
end

"""
    is_metal_element(element::String) -> Bool

Return true if the element is a metal or metalloid.
"""
function is_metal_element(element::String)::Bool
    an = get_atomic_number(element)
    an == 0 && return false
    # Alkali metals (group 1, except H)
    an in (3, 11, 19, 37, 55, 87) && return true
    # Alkaline earth metals (group 2)
    an in (4, 12, 20, 38, 56, 88) && return true
    # Transition metals (groups 3-12)
    (21 <= an <= 30) && return true
    (39 <= an <= 48) && return true
    (57 <= an <= 80) && return true
    (89 <= an <= 112) && return true
    # Post-transition metals
    an in (13, 31, 49, 50, 81, 82, 83) && return true
    # Metalloids
    an in (5, 14, 32, 33, 51, 52, 84) && return true
    # Lanthanides and actinides
    (57 <= an <= 71) && return true
    (89 <= an <= 103) && return true
    return false
end

"""
    get_element_index(element::String) -> Int

Return 1-based index of element in a canonical ordering (by atomic number).
Returns 0 for unknown elements.
"""
function get_element_index(element::String)::Int
    an = get_atomic_number(element)
    return an  # atomic number itself is the canonical index
end
