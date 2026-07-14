cmake_minimum_required(VERSION 3.10)

project(CompileProject)

# Default to Release so CMAKE_CXX_FLAGS_RELEASE (-O3 -DNDEBUG) kicks in
# for any newly-configured build. Explicit -O3 is still set per-target
# below so benchmarks stay optimized even if someone overrides BUILD_TYPE.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" ${PROJECT_SOURCE_DIR}/cmake)

# Overridable so the raw-nvcc-parity wrapper (build_ampere.sh) can emit into
# build/ where validate.R / test.R load build/mtrx.so; the default configure()
# path still uses lib/.
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/)
endif()

# Device-side -O3 for .cu files. Host-side -O3 -DNDEBUG is already
# forwarded by CMake via CMAKE_CXX_FLAGS_RELEASE when BUILD_TYPE=Release.
# -Xcompiler -fopenmp enables OpenMP in the host code (the .Call fast-path
# double<->float conversions in src/main.cu).
list(APPEND CUDA_NVCC_FLAGS -O3 -Xcompiler -fopenmp)

# Target GPU arch (sm_86 Ampere / sm_120 Blackwell). build_ampere.sh /
# build_blackwell.sh set this to match the raw-nvcc dev loop — keep in lockstep.
if(NOT DEFINED GADES_CUDA_ARCH)
    set(GADES_CUDA_ARCH sm_86 CACHE STRING "CUDA arch for the mtrx GPU lib")
endif()
list(APPEND CUDA_NVCC_FLAGS -arch=${GADES_CUDA_ARCH})

find_package(CUDA REQUIRED)
find_package(R REQUIRED)

if(CUDA_FOUND)

	# Per-metric split (ADR-0002, candidate 2): one .cu per metric + shared
	# GEMM core + thin dispatch router + runtime state. pc_linalg.cuh /
	# pc_corr_core.cuh / pc_runtime.cuh are header-only.
	cuda_add_library(
	    mtrx
	    SHARED
	    src/dispatch.cu
	    src/pc_corr_core.cu
	    src/pc_runtime.cu
	    src/pcp_dispatch.cu
	    src/register_mtrx.cu
	    src/metrics/euclidean.cu
	    src/metrics/cosine.cu
	    src/metrics/pearson.cu
	    src/metrics/spearman.cu
	    src/metrics/kendall.cu
	    src/metrics/manhattan.cu
	)
	set_target_properties(mtrx PROPERTIES PREFIX "")
	target_include_directories(mtrx PUBLIC ${R_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
	# cuBLAS is used by the Cosine/Pearson dense drivers (see src/main.cu).
	# gomp: OpenMP runtime for the .Call fast-path host conversions.
	target_link_libraries(mtrx ${R_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} gomp)
	install(
   		TARGETS mtrx
   		LIBRARY DESTINATION libs
	)

endif(CUDA_FOUND)

add_library (mtrx_cpu SHARED
    src/dispatch.cpp
    src/pc_corr_core.cpp
    src/register_mtrx_cpu.cpp
    src/metrics/euclidean.cpp
    src/metrics/cosine.cpp
    src/metrics/pearson.cpp
    src/metrics/spearman.cpp
    src/metrics/kendall.cpp
    src/metrics/manhattan.cpp
)

set_target_properties(mtrx_cpu PROPERTIES PREFIX "")
target_include_directories (mtrx_cpu PUBLIC ${R_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(mtrx_cpu PRIVATE -O3 -DNDEBUG)
find_package(OpenMP REQUIRED)
# OpenBLAS is used by the Cosine/Pearson dense CPU drivers (cblas_sgemm/ssyrk).
target_link_libraries(mtrx_cpu ${R_LIBRARIES} OpenMP::OpenMP_CXX openblas)


install(
   TARGETS mtrx_cpu
   LIBRARY DESTINATION libs
)

# Benchmark executables (ArrayFire / Armadillo) are OFF by default so the
# package configure stays clean without those deps (find_package(Armadillo)
# etc.). Enable with -DGADES_BUILD_BENCH=ON. (Full package/bench split: ADR-0002
# follow-up / candidate 6.)
option(GADES_BUILD_BENCH "Build ArrayFire/Armadillo benchmark executables" OFF)
if(GADES_BUILD_BENCH)

# ArrayFire benchmark executable
set(AF_BUILD_DIR ${PROJECT_SOURCE_DIR}/arrayfire/build)
set(AF_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/arrayfire/include)

add_executable(test_arrayfire test_arrayfire.cpp)
target_include_directories(test_arrayfire PRIVATE ${AF_INCLUDE_DIR})
target_compile_options(test_arrayfire PRIVATE -O3 -DNDEBUG)
target_link_directories(test_arrayfire PRIVATE
    ${AF_BUILD_DIR}/src/api/unified
    ${AF_BUILD_DIR}/src/backend/cuda
    ${AF_BUILD_DIR}/src/backend/cpu
)
target_link_libraries(test_arrayfire af)
set_target_properties(test_arrayfire PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    BUILD_RPATH "${AF_BUILD_DIR}/src/api/unified;${AF_BUILD_DIR}/src/backend/cuda;${AF_BUILD_DIR}/src/backend/cpu"
    INSTALL_RPATH "${AF_BUILD_DIR}/src/api/unified;${AF_BUILD_DIR}/src/backend/cuda;${AF_BUILD_DIR}/src/backend/cpu"
)

# Armadillo benchmark executable
find_package(Armadillo REQUIRED)

add_executable(test_armadillo test_armadillo.cpp)
target_include_directories(test_armadillo PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_compile_options(test_armadillo PRIVATE -O3 -DNDEBUG)
target_link_libraries(test_armadillo ${ARMADILLO_LIBRARIES})
set_target_properties(test_armadillo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

endif(GADES_BUILD_BENCH)

