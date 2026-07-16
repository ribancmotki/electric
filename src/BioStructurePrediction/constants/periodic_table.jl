"""
Periodic table data for element properties used in structure featurisation.
"""

"""
    PERIODIC_TABLE

Dict mapping element symbol to (atomic_number, atomic_mass, electronegativity, is_metal).
"""
const PERIODIC_TABLE = Dict{String,NamedTuple{(:atomic_number,:atomic_mass,:electronegativity,:is_metal),Tuple{Int,Float64,Float64,Bool}}}(
    "H"  => (atomic_number=1,  atomic_mass=1.008,   electronegativity=2.20, is_metal=false),
    "He" => (atomic_number=2,  atomic_mass=4.003,   electronegativity=0.00, is_metal=false),
    "Li" => (atomic_number=3,  atomic_mass=6.941,   electronegativity=0.98, is_metal=true),
    "Be" => (atomic_number=4,  atomic_mass=9.012,   electronegativity=1.57, is_metal=true),
    "B"  => (atomic_number=5,  atomic_mass=10.811,  electronegativity=2.04, is_metal=false),
    "C"  => (atomic_number=6,  atomic_mass=12.011,  electronegativity=2.55, is_metal=false),
    "N"  => (atomic_number=7,  atomic_mass=14.007,  electronegativity=3.04, is_metal=false),
    "O"  => (atomic_number=8,  atomic_mass=15.999,  electronegativity=3.44, is_metal=false),
    "F"  => (atomic_number=9,  atomic_mass=18.998,  electronegativity=3.98, is_metal=false),
    "Ne" => (atomic_number=10, atomic_mass=20.180,  electronegativity=0.00, is_metal=false),
    "Na" => (atomic_number=11, atomic_mass=22.990,  electronegativity=0.93, is_metal=true),
    "Mg" => (atomic_number=12, atomic_mass=24.305,  electronegativity=1.31, is_metal=true),
    "Al" => (atomic_number=13, atomic_mass=26.982,  electronegativity=1.61, is_metal=true),
    "Si" => (atomic_number=14, atomic_mass=28.086,  electronegativity=1.90, is_metal=false),
    "P"  => (atomic_number=15, atomic_mass=30.974,  electronegativity=2.19, is_metal=false),
    "S"  => (atomic_number=16, atomic_mass=32.065,  electronegativity=2.58, is_metal=false),
    "Cl" => (atomic_number=17, atomic_mass=35.453,  electronegativity=3.16, is_metal=false),
    "Ar" => (atomic_number=18, atomic_mass=39.948,  electronegativity=0.00, is_metal=false),
    "K"  => (atomic_number=19, atomic_mass=39.098,  electronegativity=0.82, is_metal=true),
    "Ca" => (atomic_number=20, atomic_mass=40.078,  electronegativity=1.00, is_metal=true),
    "Sc" => (atomic_number=21, atomic_mass=44.956,  electronegativity=1.36, is_metal=true),
    "Ti" => (atomic_number=22, atomic_mass=47.867,  electronegativity=1.54, is_metal=true),
    "V"  => (atomic_number=23, atomic_mass=50.942,  electronegativity=1.63, is_metal=true),
    "Cr" => (atomic_number=24, atomic_mass=51.996,  electronegativity=1.66, is_metal=true),
    "Mn" => (atomic_number=25, atomic_mass=54.938,  electronegativity=1.55, is_metal=true),
    "Fe" => (atomic_number=26, atomic_mass=55.845,  electronegativity=1.83, is_metal=true),
    "Co" => (atomic_number=27, atomic_mass=58.933,  electronegativity=1.88, is_metal=true),
    "Ni" => (atomic_number=28, atomic_mass=58.693,  electronegativity=1.91, is_metal=true),
    "Cu" => (atomic_number=29, atomic_mass=63.546,  electronegativity=1.90, is_metal=true),
    "Zn" => (atomic_number=30, atomic_mass=65.38,   electronegativity=1.65, is_metal=true),
    "Ga" => (atomic_number=31, atomic_mass=69.723,  electronegativity=1.81, is_metal=true),
    "Ge" => (atomic_number=32, atomic_mass=72.630,  electronegativity=2.01, is_metal=false),
    "As" => (atomic_number=33, atomic_mass=74.922,  electronegativity=2.18, is_metal=false),
    "Se" => (atomic_number=34, atomic_mass=78.971,  electronegativity=2.55, is_metal=false),
    "Br" => (atomic_number=35, atomic_mass=79.904,  electronegativity=2.96, is_metal=false),
    "Kr" => (atomic_number=36, atomic_mass=83.798,  electronegativity=3.00, is_metal=false),
    "Rb" => (atomic_number=37, atomic_mass=85.468,  electronegativity=0.82, is_metal=true),
    "Sr" => (atomic_number=38, atomic_mass=87.620,  electronegativity=0.95, is_metal=true),
    "Y"  => (atomic_number=39, atomic_mass=88.906,  electronegativity=1.22, is_metal=true),
    "Zr" => (atomic_number=40, atomic_mass=91.224,  electronegativity=1.33, is_metal=true),
    "Nb" => (atomic_number=41, atomic_mass=92.906,  electronegativity=1.60, is_metal=true),
    "Mo" => (atomic_number=42, atomic_mass=95.960,  electronegativity=2.16, is_metal=true),
    "Tc" => (atomic_number=43, atomic_mass=98.000,  electronegativity=1.90, is_metal=true),
    "Ru" => (atomic_number=44, atomic_mass=101.07,  electronegativity=2.20, is_metal=true),
    "Rh" => (atomic_number=45, atomic_mass=102.91,  electronegativity=2.28, is_metal=true),
    "Pd" => (atomic_number=46, atomic_mass=106.42,  electronegativity=2.20, is_metal=true),
    "Ag" => (atomic_number=47, atomic_mass=107.87,  electronegativity=1.93, is_metal=true),
    "Cd" => (atomic_number=48, atomic_mass=112.41,  electronegativity=1.69, is_metal=true),
    "In" => (atomic_number=49, atomic_mass=114.82,  electronegativity=1.78, is_metal=true),
    "Sn" => (atomic_number=50, atomic_mass=118.71,  electronegativity=1.96, is_metal=true),
    "Sb" => (atomic_number=51, atomic_mass=121.76,  electronegativity=2.05, is_metal=false),
    "Te" => (atomic_number=52, atomic_mass=127.60,  electronegativity=2.10, is_metal=false),
    "I"  => (atomic_number=53, atomic_mass=126.90,  electronegativity=2.66, is_metal=false),
    "Xe" => (atomic_number=54, atomic_mass=131.29,  electronegativity=2.60, is_metal=false),
    "Cs" => (atomic_number=55, atomic_mass=132.91,  electronegativity=0.79, is_metal=true),
    "Ba" => (atomic_number=56, atomic_mass=137.33,  electronegativity=0.89, is_metal=true),
    "La" => (atomic_number=57, atomic_mass=138.91,  electronegativity=1.10, is_metal=true),
    "Ce" => (atomic_number=58, atomic_mass=140.12,  electronegativity=1.12, is_metal=true),
    "W"  => (atomic_number=74, atomic_mass=183.84,  electronegativity=2.36, is_metal=true),
    "Re" => (atomic_number=75, atomic_mass=186.21,  electronegativity=1.90, is_metal=true),
    "Os" => (atomic_number=76, atomic_mass=190.23,  electronegativity=2.20, is_metal=true),
    "Ir" => (atomic_number=77, atomic_mass=192.22,  electronegativity=2.20, is_metal=true),
    "Pt" => (atomic_number=78, atomic_mass=195.08,  electronegativity=2.28, is_metal=true),
    "Au" => (atomic_number=79, atomic_mass=196.97,  electronegativity=2.54, is_metal=true),
    "Hg" => (atomic_number=80, atomic_mass=200.59,  electronegativity=2.00, is_metal=true),
    "Pb" => (atomic_number=82, atomic_mass=207.20,  electronegativity=2.33, is_metal=true),
    "Bi" => (atomic_number=83, atomic_mass=208.98,  electronegativity=2.02, is_metal=true),
)

"""
    get_atomic_number(element::String) -> Int

Return the atomic number for the given element symbol, or 0 if unknown.
"""
function get_atomic_number(element::String)::Int
    entry = get(PERIODIC_TABLE, element, nothing)
    return entry === nothing ? 0 : entry.atomic_number
end

"""
    get_atomic_mass(element::String) -> Float64

Return the atomic mass for the given element symbol, or 0.0 if unknown.
"""
function get_atomic_mass(element::String)::Float64
    entry = get(PERIODIC_TABLE, element, nothing)
    return entry === nothing ? 0.0 : entry.atomic_mass
end

"""
    is_metal_element(element::String) -> Bool

Return true if the element is a metal.
"""
function is_metal_element(element::String)::Bool
    entry = get(PERIODIC_TABLE, element, nothing)
    return entry === nothing ? false : entry.is_metal
end
