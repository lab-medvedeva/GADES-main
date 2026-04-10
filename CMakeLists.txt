cmake_minimum_required(VERSION 3.10)

project(CompileProject)

# Default to Release so CMAKE_CXX_FLAGS_RELEASE (-O3 -DNDEBUG) kicks in
# for any newly-configured build. Explicit -O3 is still set per-target
# below so benchmarks stay optimized even if someone overrides BUILD_TYPE.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/)

# Device-side -O3 for .cu files. Host-side -O3 -DNDEBUG is already
# forwarded by CMake via CMAKE_CXX_FLAGS_RELEASE when BUILD_TYPE=Release.
list(APPEND CUDA_NVCC_FLAGS -O3)

find_package(CUDA REQUIRED)
find_package(R REQUIRED)

if(CUDA_FOUND)

	cuda_add_library(
	    mtrx
	    SHARED
	    src/main.cu
	)
	set_target_properties(mtrx PROPERTIES PREFIX "")
	target_include_directories(mtrx PUBLIC ${R_INCLUDE_DIR})
	target_link_libraries(mtrx ${R_LIBRARIES})
	install(
   		TARGETS mtrx
   		LIBRARY DESTINATION libs
	)

endif(CUDA_FOUND)

add_library (mtrx_cpu SHARED src/main.cpp)

set_target_properties(mtrx_cpu PROPERTIES PREFIX "")
target_include_directories (mtrx_cpu PUBLIC ${R_INCLUDE_DIR})
target_compile_options(mtrx_cpu PRIVATE -O3 -DNDEBUG)
find_package(OpenMP REQUIRED)
target_link_libraries(mtrx_cpu ${R_LIBRARIES} OpenMP::OpenMP_CXX)


install(
   TARGETS mtrx_cpu
   LIBRARY DESTINATION libs
)

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

