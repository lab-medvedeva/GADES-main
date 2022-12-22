cmake_minimum_required(VERSION 3.10)

project(CompileProject)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" ${PROJECT_SOURCE_DIR}/lib/cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
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
else()
	add_library (mtrx_cpu SHARED src/main.cpp)

	set_target_properties(mtrx_cpu PROPERTIES PREFIX "")
	target_include_directories (mtrx_cpu PUBLIC ${R_INCLUDE_DIR})
	target_link_libraries(mtrx_cpu ${R_LIBRARIES})
endif(CUDA_FOUND)
#cuda_add_library(
#    mtrx
#    SHARED
#    main.cu
#)

