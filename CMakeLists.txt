#Command line options
#-DUSE_TBB=ON/OFF  (default off)
#-DUSE_OMP=ON/OFF  (default off)
#-DUSE_SSE=ON/OFF  (default on)
#-DUSE_AVX=ON/OFF  (default on)

# ----------------------------------------------------------------------------
#   Basic Configuration
# ----------------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(Bingonet)
set(PROJECT_VERSION "0.0.1")
string(REGEX MATCHALL "[0-9]" PROJECT_VERSION_PARTS "${PROJECT_VERSION}")
list(GET PROJECT_VERSION_PARTS 0 PROJECT_VERSION_MAJOR)
list(GET PROJECT_VERSION_PARTS 1 PROJECT_VERSION_MINOR)
list(GET PROJECT_VERSION_PARTS 2 PROJECT_VERSION_PATCH)
set(PROJECT_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

#------------------------------------------------------
# Build type
#------------------------------------------------------

IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

#------------------------------------------------------
# Lib Names and Dirs
#------------------------------------------------------

# Postfix of so's:
set(PROJECT_DLLVERSION)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_INSTALL_PREFIX}/lib/cmake/ /usr/lib/cmake)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

OPTION(USE_TBB 	"Set to ON to use TBB" OFF)
OPTION(USE_OMP 	"Set to ON to use OMP" OFF)
OPTION(USE_SSE 	"Set to ON to use sse" ON)
OPTION(USE_AVX 	"Set to ON to use avx" ON)
OPTION(BUILD_TESTS "Set to ON to build tests" OFF)


# ----------------------------------------------------------------------------
#   Find Dependencies
# ----------------------------------------------------------------------------

IF(USE_TBB)
    find_package(TBB REQUIRED)
    add_definitions(-DCNN_USE_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})
    SET(REQUIRED_LIBRARIES  ${REQUIRED_LIBRARIES} ${TBB_LIBRARIES}  )
ELSEIF(USE_OMP)
find_package(OpenMP  REQUIRED)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions( -DCNN_USE_OMP)
ENDIF()
# ----------------------------------------------------------------------------
# COMPILER OPTIONS
# ----------------------------------------------------------------------------


IF(USE_SSE)
    add_definitions(-DCNN_USE_SSE)
    SET(EXTRA_C_FLAGS  " ${EXTRA_C_FLAGS} -msse3 ")
ENDIF()
IF(USE_AVX)
    add_definitions(-DCNN_USE_AVX)
    SET(EXTRA_C_FLAGS  "${EXTRA_C_FLAGS} -mavx ")
ENDIF()
IF((NOT USE_TBB) AND (NOT USE_OMP))
    SET(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -pthread ")
ENDIF()
SET(EXTRA_C_FLAGS " ${EXTRA_C_FLAGS} -std=c++11 -Wall -Wpedantic")
SET(EXTRA_C_FLAGS_RELEASE " -O3 ")
SET(EXTRA_C_FLAGS_DEBUG " -g3 ")



    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}")
    SET(CMAKE_CXX_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG}")



#------------------------------------------------
# TARGETS
#------------------------------------------------
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR} )
LINK_LIBRARIES(${REQUIRED_LIBRARIES} )

SET( Bingonet_hrds bingonet/activations/bingo_activation_function.h  bingonet/layers/bingo_fc_layer.h  bingonet/layers/bingo_conv_layer.h  bingonet/layers/bingo_layer.h  bingonet/lossfunctions/bingo_loss_function.h  bingonet/optimizers/bingo_optimizer.h  bingonet/util/bingo_product.h  bingonet/util/bingo_util.h  bingonet/config.h  bingonet/layers/bingo_input_layer.h bingonet/layers/bingo_layers.h  bingonet/bingo_network.h  bingonet/bingonet.h  bingonet/util/bingo_weight_init.h)


SET(Bingonet_test_headers test/test_bingo_fc_layer.h test/test_bingo_network.h test/picotest/picotest.h)



IF(BUILD_TESTS)
    ADD_EXECUTABLE(Bingonet_test test/test.cpp ${Bingonet_hrds} ${Bingonet_test_headers})
ENDIF()





# ----------------------------------------------------------------------------
# display status message for important variables
# ----------------------------------------------------------------------------
message( STATUS )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
message( STATUS "General configuration for ${PROJECT_NAME} ${PROJECT_VERSION}")
MESSAGE( STATUS "-------------------------------------------------------------------------------" )


message( STATUS "BUILD_EXAMPLES=${BUILD_EXAMPLES} BUILD_TESTS=${BUILD_TESTS} USE_TBB=${USE_TBB} USE_OMP=${USE_OMP} USE_SSE=${USE_SSE} USE_AVX=${USE_AVX}")


MESSAGE( STATUS "-------------------------------------------------------------------------------" )

message( STATUS )
message("    Built as dynamic libs?:" ${BUILD_SHARED_LIBS})
message("    Compiler:"                   "${CMAKE_COMPILER}"   "${CMAKE_CXX_COMPILER}")
message( STATUS "C++ flags (Release):       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message( STATUS "C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message( STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message( STATUS "CMAKE_BINARY_DIR:         ${CMAKE_BINARY_DIR}")
message( STATUS "OpenCV libs:              ${OpenCV_LIBS}")

MESSAGE( STATUS )
MESSAGE( STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}" )
MESSAGE( STATUS "REQUIRED_LIBRARIES = ${REQUIRED_LIBRARIES}" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
MESSAGE( STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )
MESSAGE( STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}")
MESSAGE( STATUS )
MESSAGE( STATUS )
MESSAGE( STATUS "Change a value with: cmake -D<Variable>=<Value>" )
MESSAGE( STATUS )
