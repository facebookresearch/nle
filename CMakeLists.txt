cmake_minimum_required(VERSION 3.12)
file(STRINGS "version.txt" NLE_VERSION)
project(nle VERSION ${NLE_VERSION})
message(STATUS "Building nle backend version: ${NLE_VERSION}")

# This is set to build_lib in build_ext, since we use it to export the python
# interfaces into the right package.
set(PYTHON_SRC_PARENT
    ${nle_SOURCE_DIR}
    CACHE STRING "Directory containing the nle package files")

# TODO(NN): we should be at least XDG compliant
set(HACKDIR
    "$ENV{HOME}/nethackdir"
    CACHE STRING "Configuration files for nethack")

# Playground vars
set(VARDIR ${HACKDIR})
set(INSTDIR ${HACKDIR})

add_compile_definitions(
  GCC_WARN
  NOCLIPPING
  NOMAIL
  NOTPARMDECL
  HACKDIR="${HACKDIR}"
  DEFAULT_WINDOW_SYS="tty"
  DLB)

set(NLE_SRC ${nle_SOURCE_DIR}/src)
set(NLE_INC ${nle_SOURCE_DIR}/include)
set(NLE_DAT ${nle_SOURCE_DIR}/dat)
set(NLE_UTIL ${nle_SOURCE_DIR}/util)

set(NLE_SRC_GEN ${nle_BINARY_DIR}/src)
set(NLE_INC_GEN ${nle_BINARY_DIR}/include)
set(NLE_DAT_GEN ${nle_BINARY_DIR}/dat)
set(NLE_UTIL_GEN ${nle_BINARY_DIR}/util)
message(STATUS ${nle_BINARY_DIR})
message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(util)
add_subdirectory(dat)
add_subdirectory(win/rl)

file(
  GLOB
  NLEHACK_SRC
  "src/*.c"
  "sys/share/posixregex.c"
  "sys/share/ioctl.c"
  "sys/unix/unixunix.c"
  "sys/share/unixtty.c"
  "sys/unix/unixres.c"
  "sys/unix/unixmain.c"
  "win/tty/*.c")

add_executable(nethack ${NLEHACK_SRC})
add_dependencies(nethack nleutil winrl nledat)

target_include_directories(nethack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                          ${NLE_INC_GEN})

find_library(ZMQ zmq)
find_library(CURSES ncurses)
target_link_libraries(nethack PRIVATE ${ZMQ} ${CURSES} winrl)

install(TARGETS nethack)
install(TARGETS nethack DESTINATION ${INSTDIR})

install(DIRECTORY DESTINATION ${VARDIR})
install(DIRECTORY DESTINATION ${VARDIR}/save)
install(DIRECTORY DESTINATION ${INSTDIR})
