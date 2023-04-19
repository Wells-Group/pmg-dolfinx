cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME vector-update C CXX)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Ofast -march=native -stdlib=libstdc++ --gcc-toolchain=/opt/cray/pe/gcc/11.2.0/snos ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -DNDEBUG -Wall")

option(profiling          "Enable ROCm scoped profiling"                                  off)

# ------------------------------------------------------------------------------
# General configuration
find_package(MPI)
find_package(DOLFINX REQUIRED)
find_package(Boost 1.70 REQUIRED program_options)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


# Set the GPU to compile for
set(GPU_TARGETS "gfx90a" CACHE STRING "GPU targets to compile for")

add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py
  VERBATIM
  DEPENDS poisson.py
  COMMENT "Compile poisson.py using FFCx"
)

# If using ROCm and HIP
# On ROCm rocThrust requires rocPRIM
find_package(hip REQUIRED)
find_package(rocprim REQUIRED CONFIG)
find_package(rocthrust REQUIRED CONFIG)
if(profiling)
   find_library(ROCTRACER_LIBRARY NAMES roctracer64)
        find_library(ROCTX64_LIBRARY NAMES roctx64)
        find_library(ROCM_SMI64_LIBRARY NAMES rocm_smi64)
        add_definitions(-DROCM_TRACING=1)
        add_definitions(-DROCM_SMI=1)
endif()

add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_BINARY_DIR}/poisson.c)

message(WARNING ${CMAKE_CURRENT_SOURCE_DIR})
include_directories("../../")
# include_directories(${MPI_CXX_INCLUDE_DIRS})
# include_directories(${MPI_C_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} dolfinx roc::rocthrust Boost::program_options)
if(profiling)
        target_link_libraries(${PROJECT_NAME} ${ROCTX64_LIBRARY} ${ROCTRACER_LIBRARY} ${ROCM_SMI64_LIBRARY})
endif()

