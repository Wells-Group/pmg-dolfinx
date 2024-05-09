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

set(PROJECT_NAME tests C CXX)
project(${PROJECT_NAME})

message(STATUS "${Green}Compiling for AMD GPUs${ColourReset}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Ofast -march=native -stdlib=libstdc++ --gcc-toolchain=/opt/cray/pe/gcc/11.2.0/snos ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -DNDEBUG -Wall")


# ------------------------------------------------------------------------------
# General configuration
find_package(MPI)
find_package(DOLFINX REQUIRED)
find_package(Boost 1.70 REQUIRED program_options)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py
  VERBATIM
  DEPENDS poisson.py
  COMMENT "Compile poisson.py using FFCx"
)


# Set the GPU to compile for
set(GPU_TARGETS "gfx90a" CACHE STRING "GPU targets to compile for")

# If using ROCm and HIP
# On ROCm rocThrust requires rocPRIM
find_package(hip REQUIRED)
find_package(rocprim REQUIRED CONFIG)
find_package(rocthrust REQUIRED CONFIG)
find_package(hipsparse REQUIRED CONFIG)


add_definitions(-DUSE_HIP=1)
add_executable(${PROJECT_NAME} test_csr.cpp ${CMAKE_CURRENT_BINARY_DIR}/poisson.c)

include_directories("../")

target_link_libraries(${PROJECT_NAME} dolfinx roc::rocthrust roc::hipsparse Boost::program_options)