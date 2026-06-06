cmake_minimum_required(VERSION 3.21)
project(_sentencepiece_bpe CXX)

# Find Python3 first so Python3_EXECUTABLE is available for the pybind11 probe.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate pybind11 from the pip-installed package unless the caller overrides.
if(NOT DEFINED pybind11_DIR OR pybind11_DIR STREQUAL "")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pb11_result
        ERROR_QUIET
    )
    if(NOT _pb11_result EQUAL 0 OR pybind11_DIR STREQUAL "")
        message(FATAL_ERROR
            "Could not run 'python -m pybind11 --cmakedir'.\n"
            "Install pybind11 in the active environment: pip install pybind11>=2.12\n"
            "Or pass -Dpybind11_DIR=<path> to cmake.")
    endif()
    message(STATUS "pybind11 cmake dir (from pip): ${pybind11_DIR}")
endif()

find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(_sentencepiece_bpe
    src/tokenizer/sentencepiece_bpe.cpp
)

target_compile_features(_sentencepiece_bpe PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(_sentencepiece_bpe PRIVATE /O2 /utf-8)
else()
    target_compile_options(_sentencepiece_bpe PRIVATE -O2)
endif()

# Output .pyd directly into src/tokenizer/ for both single- and multi-config generators.
set(_output_dir "${CMAKE_SOURCE_DIR}/src/tokenizer")
set_target_properties(_sentencepiece_bpe PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY          "${_output_dir}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG    "${_output_dir}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE  "${_output_dir}"
    RUNTIME_OUTPUT_DIRECTORY          "${_output_dir}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG    "${_output_dir}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE  "${_output_dir}"
)
