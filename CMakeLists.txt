cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(CompileProject)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/)

option(WITH_TSAN "Compile with thread sanitizer" OFF)
if (WITH_TSAN)
  add_compile_options(-fsanitize=thread)
  add_link_options(-fsanitize=thread)
endif()

option(WITH_ASAN "Compile with address sanitizer" OFF)
if (WITH_ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

set(HWY_ENABLE_TESTS OFF)
add_subdirectory(deps/highway/)

add_library(gades2_templated INTERFACE)
target_include_directories(gades2_templated INTERFACE ${PROJECT_SOURCE_DIR}/include/private)
target_link_libraries(gades2_templated INTERFACE hwy)
target_compile_options(gades2_templated INTERFACE -march=native )

add_library(GADES2 SHARED src/GADES.cpp)
target_link_libraries(GADES2 PRIVATE gades2_templated)
target_include_directories(GADES2 PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Running a Debug build")
  include(FetchContent)
  FetchContent_Declare(
    GTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
  )
  FetchContent_MakeAvailable(GTest)

  add_executable(gades_test tests/test.cpp)
  target_link_libraries(
    gades_test
    GTest::gtest_main
    GADES2
  )
  include(GoogleTest)
  gtest_discover_tests(gades_test)
endif()

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
