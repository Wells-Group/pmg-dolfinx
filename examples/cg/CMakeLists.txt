cmake_minimum_required(VERSION 3.20)

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

set(PROJECT_NAME cg C CXX)
project(${PROJECT_NAME})

option(profiling          "Enable ROCm scoped profiling"                                  off)
option(omnitrace	  "Enable AMD omnitrace profiling"				  off)
option(cpu		  "Build for CPUs"						  off)
option(nvidia		  "Build for NVidia GPUs"					  off)
option(amd		  "Build for AMD GPUs"						  on)

if(cpu AND nvidia OR cpu AND amd)
  message(FATAL_ERROR "You cannot select both a cpu build and a GPU (amd or nvidia) build. Only select one of the three options: cpu, nvidia, amd")
endif()
if(nvidia AND amd)
  message(FATAL_ERROR "You cannot select both amd and nvidia GPU builds. Only select one of the three options: cpu, nvidia, amd")
endif()


if(omnitrace AND NOT profiling)
  set(profiling on)
endif()
message(STATUS "${Green}Profiling is ${profiling}${ColourReset}")
message(STATUS "${Green}Omnitrace labelling ${omnitrace}${ColourReset}")
if(cpu)
  message(STATUS "${Green}Compiling for CPUs${ColourReset}")
elseif(nvidia)
  message(STATUS "${Green}Compiling for NVIDIA GPUs${ColourReset}")
elseif(amd)
  message(STATUS "${Green}Compiling for AMD GPUs${ColourReset}")
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
if(omnitrace)
	if(NOT DEFINED OMNITRACE_DIR)
		set(OMNITRACE_DIR "/usr")
	endif()
	find_path(OMNITRACE_INCLUDE omnitrace/user.h "${OMNITRACE_DIR}/include")
	include_directories(${OMNITRACE_INCLUDE})
	find_library(OMNITRACE_LIBRARY NAMES omnitrace HINTS "${OMNITRACE_DIR}/lib")
	find_library(OMNITRACE_USER_LIBRARY NAMES omnitrace-user HINTS "${OMNITRACE_DIR}/lib")	
	add_definitions(-DOMNITRACE=1)
endif()
if(amd)
        add_definitions(-DAMD_GPU=1)
elseif(nvidia)
	add_definitions(-DNVIDIA_GPU=1)
elseif(cpu)
	add_definitions(-DCPU_ONLY=1)
endif()
add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_BINARY_DIR}/poisson.c)

include_directories("../../")

target_link_libraries(${PROJECT_NAME} dolfinx roc::rocthrust Boost::program_options)
if(profiling)
	target_link_libraries(${PROJECT_NAME} ${ROCTX64_LIBRARY} ${ROCTRACER_LIBRARY} ${ROCM_SMI64_LIBRARY})
endif()
if(omnitrace)
	target_link_libraries(${PROJECT_NAME} ${OMNITRACE_USER_LIBRARY} ${OMNITRACE_LIBRARY})
endif()
