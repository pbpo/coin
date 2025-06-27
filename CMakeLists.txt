cmake_minimum_required(VERSION 3.20)
project(HIPCANBert CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17) # For .hip.cpp files, ensure consistency if needed
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# ROCm/HIP path (User might need to set ROCM_PATH environment variable)
if(NOT DEFINED ENV{ROCM_PATH} AND NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
    message(STATUS "ROCM_PATH not set, defaulting to /opt/rocm. Override if necessary.")
else()
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    endif()
    message(STATUS "Using ROCM_PATH: ${ROCM_PATH}")
endif()

set(CMAKE_MODULE_PATH ${ROCM_PATH}/hip/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${ROCM_PATH} ${CMAKE_PREFIX_PATH}) # Helps find_package for ROCm components

# Enable HIP language
enable_language(HIP)

# rocBLAS
find_package(rocblas REQUIRED)
if(rocblas_FOUND)
    message(STATUS "Found rocBLAS: ${rocblas_LIBRARIES}")
else()
    message(FATAL_ERROR "rocBLAS not found! Ensure ROCM_PATH is correct and rocBLAS is installed.")
endif()

# nlohmann_json via FetchContent
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3 # Using a specific stable tag
)
FetchContent_MakeAvailable(nlohmann_json)

# Include directories
include_directories(
    src
    ${nlohmann_json_SOURCE_DIR}/include # For nlohmann/json.hpp
    ${ROCM_PATH}/include # General ROCm includes
    ${rocblas_INCLUDE_DIRS}
)

# --- Source Files ---
# Core library (will be compiled as part of executables needing them)
file(GLOB CORE_SOURCES src/core/*.cpp)
file(GLOB CORE_HEADERS src/core/*.hpp)

# Utils library
file(GLOB UTILS_SOURCES src/utils/*.cpp)
file(GLOB UTILS_HEADERS src/utils/*.hpp)

# Models (HIP sources)
file(GLOB MODELS_HIP_SOURCES src/models/*.hip.cpp)
file(GLOB MODELS_HEADERS src/models/*.hpp)

# --- Executables ---

# Utility Scripts
add_executable(aggregate_data src/scripts/aggregate_data.cpp ${UTILS_SOURCES})
target_link_libraries(aggregate_data PRIVATE nlohmann_json::nlohmann_json) # If json is needed

add_executable(split_data src/scripts/split_data.cpp ${UTILS_SOURCES})
target_link_libraries(split_data PRIVATE nlohmann_json::nlohmann_json) # If json is needed

add_executable(build_vocab src/scripts/build_vocab.cpp ${CORE_SOURCES} ${UTILS_SOURCES})
target_link_libraries(build_vocab PRIVATE nlohmann_json::nlohmann_json)

# Main Training Executable
add_executable(pretrain src/scripts/pretrain.cpp ${CORE_SOURCES} ${UTILS_SOURCES} ${MODELS_HIP_SOURCES})
target_link_libraries(pretrain PRIVATE rocblas::rocblas nlohmann_json::nlohmann_json)

# Full Training Runner (might just be a simple script orchestrator)
add_executable(run_full_training src/scripts/run_full_training.cpp)

# --- HIP specific flags (optional, for target architecture) ---
# Example for gfx90a (MI200). User should adapt for 7900XTX (gfx1100)
# list(APPEND CMAKE_HIP_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100") # Add more as needed
# For RX 7900 XTX, it's gfx1100. ROCm usually autodetects or compiles for a broader set.
# Explicitly setting can sometimes improve compile times or ensure compatibility.
# set(AMDGPU_TARGETS "gfx1100" CACHE STRING "AMDGPU target architecture(s)")
# set(CMAKE_HIP_ARCHITECTURES ${AMDGPU_TARGETS})

message(STATUS "Project configured. Build with 'cmake --build <build_dir>' or 'make' in build dir.")

# --- Installation (Optional) ---
# install(TARGETS aggregate_data split_data build_vocab pretrain run_full_training
#         RUNTIME DESTINATION bin)
# install(FILES ${CORE_HEADERS} ${UTILS_HEADERS} ${MODELS_HEADERS}
#         DESTINATION include/hip_can_bert)

# --- Testing (Optional, with CTest) ---
# enable_testing()
# add_test(NAME AggregateDataTest COMMAND aggregate_data)
# add_test(NAME BuildVocabTest COMMAND build_vocab --args_if_needed)
# ... etc.
