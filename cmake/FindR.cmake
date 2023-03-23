if(${CMAKE_VERSION} VERSION_GREATER "3.11.0")
    cmake_policy(SET CMP0074 NEW)
endif()

find_path(
    R_INCLUDE_DIR NAMES R.h
    HINTS
        $ENV{R_ROOT}/include
        ${R_ROOT}/include
        /usr/share/R/include
    REQUIRED
)

find_library(
    R_LIBRARIES
    NAMES R # libfc.so
    HINTS
        $ENV{R_ROOT}/include
        ${R_ROOT}/lib
        /usr/share/R/lib
    REQUIRED
)
set(R_VERSION 4.0)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(R
    FOUND_VAR R_FOUND
    REQUIRED_VARS
        R_LIBRARIES
        R_INCLUDE_DIR
    VERSION_VAR R_VERSION
)

