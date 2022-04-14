cmake_minimum_required(VERSION 3.15)
file(STRINGS "version.txt" NLE_VERSION)
project(nle VERSION ${NLE_VERSION})

if(CMAKE_BUILD_TYPE MATCHES Debug)
  message("Debug build.")
  # Unclear if this is even necessary. `dsymutil rlmain -o rlmain.dSYM` seems to
  # have done the trick.
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
  set(CMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym")

  if(0)
    # address sanitizer.
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_C_FLAGS_DEBUG
        "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG
        "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address"
    )
  endif()
  if(MSVC)
    add_compile_options(/W4)
  else()
    add_compile_options(-Wall)
  endif()
elseif(CMAKE_BUILD_TYPE MATCHES Release)
  message("Release build.")
else()
  message("Some other build type.")
endif()

message(STATUS "Building nle backend version: ${NLE_VERSION}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# We use this to decide where the root of the nle/ package is. Normally it
# shouldn't be needed, but sometimes (e.g. when using setuptools) we are
# generating some of the files outside of the original package path.
set(PYTHON_SRC_PARENT
    ${nle_SOURCE_DIR}
    CACHE STRING "Directory containing the nle package files")

set(HACKDIR
    "$ENV{HOME}/nethackdir.nle"
    CACHE STRING "Configuration files for nethack")

message(STATUS "HACKDIR set to: ${HACKDIR}")

# Playground vars
set(VARDIR ${HACKDIR})
set(INSTDIR ${HACKDIR})

add_compile_definitions(
  GCC_WARN
  NOCLIPPING
  NOMAIL
  NOTPARMDECL
  HACKDIR="${HACKDIR}"
  DEFAULT_WINDOW_SYS="rl"
  DLB
  NOCWD_ASSUMPTIONS
  NLE_USE_TILES)

option(USE_SEEDING "Use seeding support in NLE" ON)

if(USE_SEEDING)
  add_compile_definitions(NLE_ALLOW_SEEDING)
  message("Seeding enabled.")
else()
  message("Seeding disabled.")
endif()

set(NLE_SRC ${nle_SOURCE_DIR}/src)
set(NLE_INC ${nle_SOURCE_DIR}/include)
set(NLE_DAT ${nle_SOURCE_DIR}/dat)
set(NLE_UTIL ${nle_SOURCE_DIR}/util)
set(NLE_DOC ${nle_SOURCE_DIR}/doc)
set(NLE_SSYS ${nle_SOURCE_DIR}/sys/share)
set(NLE_WIN ${nle_SOURCE_DIR}/win)

set(NLE_SRC_GEN ${nle_BINARY_DIR}/src)
set(NLE_INC_GEN ${nle_BINARY_DIR}/include)
set(NLE_DAT_GEN ${nle_BINARY_DIR}/dat)
set(NLE_UTIL_GEN ${nle_BINARY_DIR}/util)

set(CMAKE_INSTALL_MESSAGE LAZY) # Don't tell us about up-to-date files.

# EXCLUDE_FROM_ALL: Don't install this static library into /usr/local.
add_subdirectory(third_party/deboost.context EXCLUDE_FROM_ALL)
add_subdirectory(util)
add_subdirectory(dat)

file(
  GLOB
  NETHACK_SRC
  "src/*.c"
  "sys/share/posixregex.c"
  "sys/share/ioctl.c"
  "sys/unix/unixunix.c"
  "sys/unix/unixmain.c"
  "sys/unix/unixres.c"
  "win/tty/*.c"
  "win/rl/winrl.cc")

# terminal emulator library
add_library(tmt STATIC "third_party/libtmt/tmt.c")
set_target_properties(tmt PROPERTIES C_STANDARD 11)

# libnethack library
add_library(nethack SHARED ${NETHACK_SRC})
add_dependencies(nethack util dat)
set_target_properties(nethack PROPERTIES CXX_STANDARD 14 SUFFIX ".so")
target_include_directories(
  nethack
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${NLE_INC_GEN} /usr/local/include
         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtmt)
target_link_directories(nethack PUBLIC /usr/local/lib)

# Careful with -DMONITOR_HEAP: Ironically, it fails to fclose FILE* heaplog.
# target_compile_definitions(nethack PUBLIC "$<$<CONFIG:DEBUG>:MONITOR_HEAP>")

target_link_libraries(nethack PUBLIC m fcontext bz2 tmt)

# dlopen wrapper library
add_library(nethackdl STATIC "sys/unix/nledl.c")
target_include_directories(nethackdl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nethackdl PUBLIC dl)

# rlmain C++ (test) binary
add_executable(rlmain "sys/unix/rlmain.cc")
set_target_properties(rlmain PROPERTIES CXX_STANDARD 11)
target_link_libraries(rlmain PUBLIC nethackdl)
target_include_directories(rlmain PUBLIC ${NLE_INC_GEN})
add_dependencies(rlmain util) # For pm.h.

# pybind11 core python library.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)
pybind11_add_module(
  _pynethack
  win/rl/pynethack.cc
  src/monst.c
  src/decl.c
  src/drawing.c
  src/objects.c
  $<TARGET_OBJECTS:tile>)
target_link_libraries(_pynethack PUBLIC nethackdl)
set_target_properties(_pynethack PROPERTIES CXX_STANDARD 14)
target_include_directories(_pynethack PUBLIC ${NLE_INC_GEN})
add_dependencies(_pynethack util) # For pm.h.

# ttyrec converter library
add_library(
  converter STATIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/converter/converter.c
                   ${CMAKE_CURRENT_SOURCE_DIR}/third_party/converter/stripgfx.c)
target_include_directories(
  converter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtmt
                   ${CMAKE_CURRENT_SOURCE_DIR}/third_party/converter)
target_link_libraries(converter PUBLIC bz2 tmt)
if(CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(converter PRIVATE -Wall -Wextra -pedantic -Werror)
endif()

# ttyrec reader executable
add_executable(ttyrec_reader EXCLUDE_FROM_ALL "third_party/converter/reader.c")
target_link_libraries(ttyrec_reader PUBLIC converter)
target_include_directories(
  ttyrec_reader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/converter)

pybind11_add_module(_pyconverter third_party/converter/pyconverter.cc)
target_link_libraries(_pyconverter PUBLIC converter)
set_target_properties(_pyconverter PROPERTIES CXX_STANDARD 14)
target_include_directories(
  _pyconverter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/converter)
