cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME cg C CXX)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Ofast -march=native -stdlib=libstdc++ --gcc-toolchain=/opt/cray/pe/gcc/11.2.0/snos ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -DNDEBUG -Wall")


# ------------------------------------------------------------------------------
# General configuration
find_package(MPI)
find_package(DOLFINX REQUIRED)

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

add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_BINARY_DIR}/poisson.c)

include_directories("../../")

target_link_libraries(${PROJECT_NAME} dolfinx roc::rocthrust)
