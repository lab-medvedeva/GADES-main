cmake_minimum_required(VERSION 3.18)
project(gades LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
set(STUB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc/r_stubs")

# ─── CPU backend (always built) ─────────────────────────────────────────────
find_package(OpenMP)

find_library(OPENBLAS_LIB NAMES openblas
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
          /usr/local/opt/openblas/lib /opt/OpenBLAS/lib
)
find_path(OPENBLAS_INCLUDE cblas.h
    PATHS /usr/include /usr/include/openblas /usr/include/x86_64-linux-gnu
          /usr/local/include /usr/local/opt/openblas/include
          /opt/OpenBLAS/include
)

if(NOT OPENBLAS_LIB)
    message(FATAL_ERROR "OpenBLAS not found. Install with: sudo apt install libopenblas-dev")
endif()

# Per-metric split: main.cpp was replaced by one .cpp per metric +
# the shared cuBLAS/cblas GEMM core (pc_corr_core). The Python api_cpu layer
# only calls the metric drivers, so dispatch.cpp / register_mtrx_cpu.cpp (the
# R .Call/registration glue) are intentionally left out here.
add_library(_gades_cpu SHARED
    ${SRC_DIR}/pc_corr_core.cpp
    ${SRC_DIR}/metrics/euclidean.cpp
    ${SRC_DIR}/metrics/cosine.cpp
    ${SRC_DIR}/metrics/pearson.cpp
    ${SRC_DIR}/metrics/spearman.cpp
    ${SRC_DIR}/metrics/kendall.cpp
    ${SRC_DIR}/metrics/manhattan.cpp
    csrc/api_cpu.cpp
)
set_target_properties(_gades_cpu PROPERTIES PREFIX "")
target_include_directories(_gades_cpu PRIVATE
    ${STUB_DIR}            # R.h / Rinternals.h stubs (must come before system)
    ${SRC_DIR}             # pc_corr_core_cpu.h / pc_runtime_cpu.h / pc_linalg_cpu.h
    ${OPENBLAS_INCLUDE}
)
target_compile_options(_gades_cpu PRIVATE -O3 -DNDEBUG -fopenmp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_gades_cpu PRIVATE OpenMP::OpenMP_CXX ${OPENBLAS_LIB})
else()
    target_link_libraries(_gades_cpu PRIVATE ${OPENBLAS_LIB} gomp)
endif()
install(TARGETS _gades_cpu LIBRARY DESTINATION gades)


# ─── GPU backend (optional — requires CUDA toolkit) ─────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    find_package(CUDAToolkit REQUIRED)

    # Per-metric split: main.cu → dispatch (provides check_gpu) +
    # shared GEMM core + runtime state + sparse-Kendall launch policy + one .cu
    # per metric. register_mtrx.cu (R registration) is not needed for the
    # ctypes-facing Python lib.
    add_library(_gades_gpu SHARED
        ${SRC_DIR}/dispatch.cu
        ${SRC_DIR}/pc_corr_core.cu
        ${SRC_DIR}/pc_runtime.cu
        ${SRC_DIR}/pcp_dispatch.cu
        ${SRC_DIR}/metrics/euclidean.cu
        ${SRC_DIR}/metrics/cosine.cu
        ${SRC_DIR}/metrics/pearson.cu
        ${SRC_DIR}/metrics/spearman.cu
        ${SRC_DIR}/metrics/kendall.cu
        ${SRC_DIR}/metrics/manhattan.cu
        csrc/api_gpu.cu
    )
    set_target_properties(_gades_gpu PROPERTIES
        PREFIX ""
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_include_directories(_gades_gpu PRIVATE
        ${STUB_DIR}        # R.h / Rinternals.h stubs
        ${SRC_DIR}         # pc_runtime.cuh / pc_linalg.cuh / pc_corr_core.cuh
    )
    target_compile_options(_gades_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xcompiler=-fopenmp>
    )
    target_link_libraries(_gades_gpu PRIVATE
        CUDA::cublas
        gomp    # OpenMP runtime for host-side conversions
    )
    install(TARGETS _gades_gpu LIBRARY DESTINATION gades)

    message(STATUS "CUDA found — building GPU backend (_gades_gpu.so)")
else()
    message(STATUS "CUDA not found — GPU backend will not be built")
endif()
