cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


project(CompileProject)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/)
set(HWY_ENABLE_TESTS OFF)
add_subdirectory(deps/highway/)
add_library(gades2_templated INTERFACE)
target_include_directories(gades2_templated INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gades2_templated INTERFACE hwy)

add_executable(gades2_compile src/compile.cpp)
target_link_libraries(gades2_compile PRIVATE gades2_templated)

option(WITH_CUDA "Install With CUDA" OFF)
find_package(R REQUIRED)

if (WITH_CUDA)
    message("Building with CUDA required")
    find_package(CUDA REQUIRED)
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

endif(WITH_CUDA)

# add_library (mtrx_cpu SHARED src/main.cpp)

# set_target_properties(mtrx_cpu PROPERTIES PREFIX "")
# target_include_directories (mtrx_cpu PUBLIC ${R_INCLUDE_DIR})
# target_link_libraries(mtrx_cpu ${R_LIBRARIES})

# install(
#    TARGETS mtrx_cpu
#    LIBRARY DESTINATION libs
# )

