cmake_minimum_required(VERSION 3.20)
project(hoji LANGUAGES CXX)

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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python")
  set(HOJI_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python"
      CACHE FILEPATH "Python interpreter used for reference dumps")
else()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(HOJI_PYTHON "${Python3_EXECUTABLE}"
      CACHE FILEPATH "Python interpreter used for reference dumps")
endif()

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
