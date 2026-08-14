cmake_minimum_required(VERSION 3.23)
project(hoji LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# So clangd/editors resolve the same includes the build uses.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Third-party (vendored, header-only)
#
# SYSTEM so -Wall -Wextra stays quiet inside 27k lines of json.hpp, and so a
# warning there can never fail our build.
# ---------------------------------------------------------------------------
add_library(hoji_third_party INTERFACE)
target_include_directories(hoji_third_party SYSTEM
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# ---------------------------------------------------------------------------
# Engine
#
#   cmake --build build            # default target (hoji)
#
# Everything except main.cc lives in hoji_core, so test binaries can link the
# engine without pulling in main().
# ---------------------------------------------------------------------------
# CONFIGURE_DEPENDS makes the build system re-glob when src/ changes, so a new
# file is picked up by `cmake --build` without a manual reconfigure.
file(GLOB_RECURSE HOJI_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

# Headers are globbed only so they show up in IDEs and in the header file set;
# the compiler finds them via the src/ include root below.
file(GLOB_RECURSE HOJI_HEADERS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

set(HOJI_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc")
set(HOJI_CORE_SOURCES ${HOJI_SOURCES})
list(REMOVE_ITEM HOJI_CORE_SOURCES "${HOJI_MAIN}")

add_library(hoji_core STATIC ${HOJI_CORE_SOURCES})

# src/ is the single include root: headers refer to each other root-relatively,
# e.g. src/model/config.h says #include "dtype.h" and #include "model/foo.h".
# PUBLIC so anything linking hoji_core inherits the same include root.
target_sources(hoji_core
  PUBLIC
    FILE_SET hoji_headers TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src"
    FILES ${HOJI_HEADERS})

target_include_directories(hoji_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

# PRIVATE: json.hpp is an implementation detail of the loader. If a public
# header ever includes it, consumers fail to compile — which is the point.
target_link_libraries(hoji_core PRIVATE hoji_third_party)

target_compile_features(hoji_core PUBLIC cxx_std_23)
target_compile_options(hoji_core PRIVATE -Wall -Wextra)

add_executable(hoji ${HOJI_MAIN})
target_link_libraries(hoji PRIVATE hoji_core)
target_compile_options(hoji PRIVATE -Wall -Wextra)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python")
  set(HOJI_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python"
      CACHE FILEPATH "Python interpreter used for reference dumps")
else()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(HOJI_PYTHON "${Python3_EXECUTABLE}"
      CACHE FILEPATH "Python interpreter used for reference dumps")
endif()

# ---------------------------------------------------------------------------
# Benchmarks
#
#   cmake --build build --target hoji_bench && ./build/hoji_bench
#
# A separate target so profiling flags stay off the engine: -g for symbols and
# -fno-omit-frame-pointer so a sampling profiler can walk the stack, neither of
# which a plain Release build gives you. -O3 is forced so the numbers mean
# something even when the tree is configured as Debug.
#
# bench/ sits outside src/, so hoji_core's glob never picks these up.
# ---------------------------------------------------------------------------
# _CPU_MATMUL_FOLDING_FACTOR is a macro, so its value is fixed per translation
# unit: one library can only expose one setting. Build one shared library per
# value; bench/bench.py loads them all and sweeps.
#
# Each library compiles cpu.cc itself rather than linking hoji_core, so the -D
# actually reaches the kernel body.
set(HOJI_BENCH_FOLDS 1 4 8 12 16)

set(HOJI_BACKEND_C "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/backend_c.cc")
set(HOJI_CPU_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cpu.cc")

# Highest optimization available, so a slow number means a slow kernel rather
# than a flag we forgot. Deliberately NOT -Ofast/-ffast-math: those enable FP
# reassociation, which would let the compiler fold the reduction itself and hide
# exactly the effect these benches exist to measure.
set(HOJI_BENCH_FLAGS
  -Wall -Wextra
  -O3 -mcpu=native -funroll-loops -ffp-contract=fast -fno-math-errno -flto
  -g -fno-omit-frame-pointer)

set(HOJI_BENCH_TARGETS)
foreach(fold IN LISTS HOJI_BENCH_FOLDS)
  set(lib "hoji_backend_f${fold}")
  add_library(${lib} SHARED ${HOJI_BACKEND_C} ${HOJI_CPU_KERNELS})
  target_include_directories(${lib} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_definitions(${lib} PRIVATE _CPU_MATMUL_FOLDING_FACTOR=${fold})
  target_compile_features(${lib} PRIVATE cxx_std_23)
  target_compile_options(${lib} PRIVATE ${HOJI_BENCH_FLAGS})
  target_link_options(${lib} PRIVATE -flto)
  list(APPEND HOJI_BENCH_TARGETS ${lib})
endforeach()

# Sweep every folding factor from Python and write a plot.
add_custom_target(bench
  COMMAND "${HOJI_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/bench/bench.py"
          --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM
  USES_TERMINAL)
add_dependencies(bench ${HOJI_BENCH_TARGETS})

# ---------------------------------------------------------------------------
# Reference tensors
#
# Dumps activations/logits from the HF implementation so kernels can be
# checked numerically against a known-good forward pass.
#
#   cmake --build build --target ref-tensors
# ---------------------------------------------------------------------------
set(HOJI_REF_MODEL "Qwen/Qwen2-0.5B-Instruct"
    CACHE STRING "HF model id to dump reference tensors from")
set(HOJI_REF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ref"
    CACHE PATH "Directory the reference tensors are written to")


set(HOJI_DUMP_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tools/dump_tensors.py")

# meta.json is written last by the script, so it stands in as the stamp for
# the whole dump: the command only re-runs if the script changes or ref/ is
# cleared (a dump downloads the model and runs a full fp32 forward pass).
add_custom_command(
  OUTPUT "${HOJI_REF_DIR}/meta.json"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${HOJI_REF_DIR}"
  COMMAND "${HOJI_PYTHON}" "${HOJI_DUMP_SCRIPT}" "${HOJI_REF_MODEL}" "${HOJI_REF_DIR}"
  DEPENDS "${HOJI_DUMP_SCRIPT}"
  COMMENT "Dumping reference tensors for ${HOJI_REF_MODEL} -> ${HOJI_REF_DIR}"
  VERBATIM
  USES_TERMINAL)

add_custom_target(ref-tensors DEPENDS "${HOJI_REF_DIR}/meta.json")

# Force a re-dump even if ref/meta.json is up to date.
add_custom_target(ref-tensors-clean
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${HOJI_REF_DIR}"
  COMMENT "Removing ${HOJI_REF_DIR}"
  VERBATIM)
