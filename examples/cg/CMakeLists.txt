cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME cg C CXX)
project(${PROJECT_NAME})

option(profiling          "Enable ROCm scoped profiling"                                  off)
option(cpu		  "Build for CPUs"						  off)
option(nvidia		  "Build for NVidia GPUs"					  off)
option(amd		  "Build for AMD GPUs"						  on)

if(cpu AND nvidia OR cpu AND amd)
  message(FATAL_ERROR "You cannot select both a cpu build and a GPU (amd or nvidia) build. Only select one of the three options: cpu, nvidia, amd")
endif()
if(nvidia AND amd)
  message(FATAL_ERROR "You cannot select both amd and nvidia GPU builds. Only select one of the three options: cpu, nvidia, amd")
endif()

message(STATUS "Profiling is ${profiling}")
if(cpu)
  message(STATUS "Compiling for CPUs")
elseif(nvidia)
  message(STATUS "Compiling for NVIDIA GPUs")
elseif(amd)
  message(STATUS "Compiling for AMD GPUs")
else()
  message(FATAL_ERROR "No target selected for the build. Please select one of the three options: cpu, nvidia, amd")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Ofast -march=native -stdlib=libstdc++ --gcc-toolchain=/opt/cray/pe/gcc/11.2.0/snos ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -DNDEBUG -Wall")


# ------------------------------------------------------------------------------
# General configuration
find_package(MPI)
find_package(DOLFINX REQUIRED)
find_package(Boost 1.70 REQUIRED program_options)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


# Set the GPU to compile for
if(amd)
  set(GPU_TARGETS "gfx90a" CACHE STRING "GPU targets to compile for")
endif()

if(nvidia)
  set(GPU_TARGETS "Ampere" CACHE STRING "GPU targets to compile for")
endif()

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

include_directories("../../")

target_link_libraries(${PROJECT_NAME} dolfinx roc::rocthrust Boost::program_options)
if(profiling)
	target_link_libraries(${PROJECT_NAME} ${ROCTX64_LIBRARY} ${ROCTRACER_LIBRARY} ${ROCM_SMI64_LIBRARY})
endif()
