cmake_minimum_required(VERSION 3.28)
project(aoc_2025)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prevent in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds not allowed. Please use: cmake -B build")
endif()

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Symlink compile_commands.json to source dir for clangd/LSP support
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    file(CREATE_LINK
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_SOURCE_DIR}/compile_commands.json"
        SYMBOLIC
    )
endif()

# Pre-create the generated directory so it exists for include paths
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")

# ==============================================================================
# Shared Flags Configuration
# ==============================================================================
# Define flags here so they stay in perfect sync between the executable and ASM step
set(SHARED_CXX_FLAGS
    -march=znver3 # build depends on a target that supports AVX2
    -O3
    -Wall -Wextra -Wpedantic
    -Wno-c23-extensions
)
set(SHARED_RELEASE_FLAGS
    # -flto
)
set(SHARED_DEBUG_FLAGS
    -g -O0 -fno-omit-frame-pointer -fsanitize=address,undefined
    # fno-sanitize-recover=all
)

# ==============================================================================
# LUT Generator
# ==============================================================================
add_executable(gen_lut src/gen_lut.cpp)
target_compile_options(gen_lut PRIVATE -O3)

set(GENERATED_LUT "${CMAKE_BINARY_DIR}/generated/digit_lut.inc")

# Generate the LUT
add_custom_command(
    OUTPUT ${GENERATED_LUT}
    COMMAND $<TARGET_FILE:gen_lut> > ${GENERATED_LUT}
    DEPENDS gen_lut
    COMMENT "Generating digit lookup table..."
    VERBATIM
)

# ==============================================================================
# Main Executable
# ==============================================================================
# Added header files explicitly so clangd and IDEs index them properly
add_executable(_2025
    src/main.cpp
    src/bench.hpp
    src/defs.hpp
    ${GENERATED_LUT}
)

# Target-specific flags
target_compile_options(_2025 PRIVATE
    ${SHARED_CXX_FLAGS}
    $<$<CONFIG:Release>:${SHARED_RELEASE_FLAGS}>
    # Add Address & Undefined Behavior sanitizers for easier AoC bug catching in Debug
    $<$<CONFIG:Debug>:${SHARED_DEBUG_FLAGS}>
)

target_compile_definitions(_2025 PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)

target_link_options(_2025 PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

# Include directories
target_include_directories(_2025 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/generated"
)

# ==============================================================================
# Assembly Generation & MCA
# ==============================================================================
add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/generated/main.s"
    COMMAND ${CMAKE_CXX_COMPILER}
        -std=c++23
        ${SHARED_CXX_FLAGS}
        ${SHARED_RELEASE_FLAGS} -DNDEBUG # Hardcoding release/optimizations for MCA profiling
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -I${CMAKE_BINARY_DIR}/generated
        #-S -masm=intel
        -S -masm=att # llvm-mca expects AT&T syntax
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
        -o ${CMAKE_BINARY_DIR}/generated/main.s
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp" ${GENERATED_LUT}
    COMMENT "Generating assembly (main.s)..."
    VERBATIM
)

# Ensure the .s file is generated every time you build
add_custom_target(generate_asm ALL DEPENDS "${CMAKE_BINARY_DIR}/generated/main.s")
