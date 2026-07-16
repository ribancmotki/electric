cmake_minimum_required(VERSION 3.18)
project(BioStructurePredictionExtensions CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Julia REQUIRED)

include_directories(${Julia_INCLUDE_DIRS})

add_library(bio_extensions SHARED
    src/extensions/geometry_ext.cpp
)

target_link_libraries(bio_extensions ${Julia_LIBRARIES})

install(TARGETS bio_extensions LIBRARY DESTINATION lib)
