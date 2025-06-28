cmake_minimum_required(VERSION 3.20)
project(HIPCANBert CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17) 
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# ROCm/HIP 경로 (사용자가 ROCM_PATH 환경 변수를 설정해야 할 수 있음)
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
set(CMAKE_PREFIX_PATH ${ROCM_PATH} ${CMAKE_PREFIX_PATH})

# HIP 언어 활성화
enable_language(HIP)

# rocBLAS 찾기
find_package(rocblas REQUIRED)
if(rocblas_FOUND)
    message(STATUS "Found rocBLAS: ${rocblas_LIBRARIES}")
else()
    message(FATAL_ERROR "rocBLAS not found! Ensure ROCM_PATH is correct and rocBLAS is installed.")
endif()

# FetchContent를 통해 nlohmann_json 가져오기
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3 # 특정 안정 태그 사용
)
FetchContent_MakeAvailable(nlohmann_json)

# 포함 디렉토리
include_directories(
    . # 현재 디렉토리를 포함하도록 수정
    ${nlohmann_json_SOURCE_DIR}/include # nlohmann/json.hpp 용
    ${ROCM_PATH}/include # 일반 ROCm 포함
    ${rocblas_INCLUDE_DIRS}
)

# --- 소스 파일 ---
# file(GLOB ...) 경로를 현재 디렉토리에 맞게 수정
file(GLOB ALL_SOURCES 
    "*.cpp"
    "*.hpp"
)

# --- 실행 파일 ---
# 모든 소스 파일을 사용하여 하나의 실행 파일 빌드 (프로젝트 구조 단순화)
add_executable(pretrain ${ALL_SOURCES})
target_link_libraries(pretrain PRIVATE rocblas::rocblas nlohmann_json::nlohmann_json)

message(STATUS "Project configured. Build with 'cmake --build <build_dir>' or 'make' in build dir.")

# --- 설치 및 테스트 (선택 사항) ---
# install(TARGETS pretrain RUNTIME DESTINATION bin)
# enable_testing()
# add_test(NAME PretrainTest COMMAND pretrain)
