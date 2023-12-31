cmake_minimum_required(VERSION 3.22.1)
project(ggml-tcl VERSION 1.0.0 LANGUAGES CXX C)
message(project: ${PROJECT_NAME})

set(TARGET ${PROJECT_NAME})
set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TCL EXACT 8.6.13 REQUIRED)  # TCL_INCLUDE_PATH TCL_LIBRARY
find_package (PkgConfig REQUIRED)
pkg_check_modules (GGML REQUIRED ggml)


message(STATUS "TCL_INCLUDE_PATH: ${TCL_INCLUDE_PATH}")
message(STATUS "TCL_LIBRARY: ${TCL_LIBRARY}")
message(STATUS "GGML_LIBRARY_DIRS: ${GGML_LIBRARY_DIRS}")
message(STATUS "GGML_INCLUDE_DIRS: ${GGML_INCLUDE_DIRS}")

#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_C_FLAGS "-g -DTCL_THREADS -DPROJECT_VERSION=${PROJECT_VERSION} ${CMAKE_CXX_FLAGS}")

add_library(${PROJECT_NAME} SHARED src/library.c
        src/tensor.c
        src/common.c
        src/context.c
        src/cgraph.c
        src/opt.c)
set_target_properties(${PROJECT_NAME}
        PROPERTIES POSITION_INDEPENDENT_CODE ON
        INSTALL_RPATH_USE_LINK_PATH ON
)

include_directories(${GGML_INCLUDE_DIRS} ${TCL_INCLUDE_PATH})
link_directories(${GGML_LIBRARY_DIRS} ${TCL_LIBRARY_PATH})
target_link_directories(${PROJECT_NAME} PRIVATE ${GGML_LIBRARY_DIRS} ${TCL_LIBRARY_PATH})
target_link_libraries(${PROJECT_NAME} PRIVATE ggml ${TCL_LIBRARY})
get_filename_component(TCL_LIBRARY_PATH "${TCL_LIBRARY}" PATH)

install(TARGETS ${TARGET}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${TARGET}${PROJECT_VERSION}
)

configure_file(pkgIndex.tcl.in pkgIndex.tcl @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pkgIndex.tcl
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${TARGET}${PROJECT_VERSION}
)
