cmake_minimum_required(VERSION 3.23)
project(hoji LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# So clangd/editors resolve the same includes the build uses.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Both backends are always compiled and linked; this only selects which one the
# unsuffixed C entry point hoji_matmul_f32 routes to. backend_c.cc also exposes
# hoji_matmul_f32_cpu and hoji_matmul_f32_metal unconditionally, so one loaded
# library can run both and compare them -- which is what bench/ does.
#
#   cmake -B build                       # hoji_matmul_f32 -> Metal (default)
#   cmake -B build -D_BACKEND_CPU=ON     # hoji_matmul_f32 -> CPU
option(_BACKEND_CPU "Route hoji_matmul_f32 to the CPU backend, not Metal" OFF)

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
# Held out of hoji_core: it needs the Metal frameworks and the compiled-in
# metallib path, and it must not be built at all when the Metal toolchain is
# missing. The Metal section below turns it into hoji_metal.
set(HOJI_METAL_HOST "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal.cc")

# Both backends are always compiled: backend::cpu and backend::metal coexist, so
# both type-check every build and one loaded library can run either. _BACKEND_CPU
# only picks which one backend_c.cc's unsuffixed hoji_matmul_f32 routes to.
set(HOJI_CORE_SOURCES ${HOJI_SOURCES})
list(REMOVE_ITEM HOJI_CORE_SOURCES "${HOJI_MAIN}" "${HOJI_METAL_HOST}")

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

# PUBLIC: backend_c.cc switches on it, and so does anything else choosing between
# backend::cpu::matmul and backend::metal::matmul.
if(_BACKEND_CPU)
  target_compile_definitions(hoji_core PUBLIC _BACKEND_CPU)
endif()

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
# Metal shaders:  .metal -> .air -> .metallib
#
# Needs the Metal toolchain, which ships separately from Xcode:
#   xcodebuild -downloadComponent MetalToolchain
#
# The .metallib is a data file, so the loader has to find it at runtime. It is
# written next to the binaries in the build tree and its path is compiled into
# hoji_metal as HOJI_METALLIB_PATH.
# ---------------------------------------------------------------------------
find_program(HOJI_METAL_CC metal
  HINTS /usr/bin /Library/Developer/CommandLineTools/usr/bin
  PATHS ENV PATH)
if(NOT HOJI_METAL_CC)
  execute_process(COMMAND xcrun --find metal
                  OUTPUT_VARIABLE HOJI_METAL_CC
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
endif()

if(HOJI_METAL_CC)
  file(GLOB_RECURSE HOJI_METAL_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/*.metal")

  # -fno-fast-math: MSL enables fast math by default, which reassociates FP and
  # would put the GPU results outside the tolerance the numpy checks use.
  # -frecord-sources keeps source in the .air for the Metal debugger.
  # -I src: kernels share headers with the C++ backends (backend/tensor_desc.h),
  # using the same src-relative include root the rest of the tree uses.
  set(HOJI_METAL_FLAGS -std=metal3.1 -O2 -fno-fast-math -frecord-sources
      -I "${CMAKE_CURRENT_SOURCE_DIR}/src")

  set(HOJI_AIR_FILES)
  foreach(shader IN LISTS HOJI_METAL_SOURCES)
    get_filename_component(stem "${shader}" NAME_WE)
    set(air "${CMAKE_CURRENT_BINARY_DIR}/shaders/${stem}.air")
    add_custom_command(
      OUTPUT "${air}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
      COMMAND xcrun -sdk macosx metal ${HOJI_METAL_FLAGS} -c "${shader}" -o "${air}"
      DEPENDS "${shader}"
      COMMENT "metal -c ${stem}.metal"
      VERBATIM)
    list(APPEND HOJI_AIR_FILES "${air}")
  endforeach()

  if(HOJI_AIR_FILES)
    set(HOJI_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/hoji.metallib")
    add_custom_command(
      OUTPUT "${HOJI_METALLIB}"
      COMMAND xcrun -sdk macosx metallib ${HOJI_AIR_FILES} -o "${HOJI_METALLIB}"
      DEPENDS ${HOJI_AIR_FILES}
      COMMENT "metallib -> hoji.metallib"
      VERBATIM)
    # ALL so a plain `cmake --build build` compiles the shaders. A kernel that
    # stopped compiling should break the build rather than wait for someone to
    # ask for the target.
    add_custom_target(hoji_shaders ALL DEPENDS "${HOJI_METALLIB}")

    # Host side. Plain C++ against metal-cpp (third_party/metal_cpp.hpp), not
    # Objective-C++, so no -fobjc-arc and no .mm. Built whenever the toolchain is
    # present, regardless of which backend is selected.
    add_library(hoji_metal STATIC ${HOJI_METAL_HOST})
    add_dependencies(hoji_metal hoji_shaders)
    target_link_libraries(hoji_metal PRIVATE hoji_third_party)
    target_include_directories(hoji_metal PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_features(hoji_metal PUBLIC cxx_std_23)
    # The loader opens this path at runtime, so it has to be absolute.
    target_compile_definitions(hoji_metal PRIVATE
      HOJI_METALLIB_PATH="${HOJI_METALLIB}")
    target_compile_options(hoji_metal PRIVATE -Wall -Wextra)
    target_link_libraries(hoji_metal PUBLIC
      "-framework Metal" "-framework Foundation")

    # PUBLIC and on hoji_core: backend_c.cc lives there and calls into it.
    # _BACKEND_METAL tells backend_c.cc the Metal entry point is linkable.
    target_link_libraries(hoji_core PUBLIC hoji_metal)
    target_compile_definitions(hoji_core PUBLIC _BACKEND_METAL)
  endif()
elseif(NOT _BACKEND_CPU)
  message(FATAL_ERROR
          "the Metal backend was selected but the metal compiler was not found."
          " Install it with: xcodebuild -downloadComponent MetalToolchain"
          " -- or configure with -D_BACKEND_CPU=ON")
else()
  message(STATUS "metal compiler not found; CPU backend only,"
                 " and hoji_matmul_f32_metal will be absent.")
endif()

# Benchmarks live in bench/ and are driven by pytest, which compiles a backend
# per (T_M, T_N, C_N) itself and caches it under TMPDIR -- see bench/conftest.py.
# Nothing here builds them, so any tile can be swept without touching CMake:
#
#   ./venv/bin/pytest bench -q
#   ./venv/bin/pytest bench -q -m perf --perf-report --tiles 4x4,4x8

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
