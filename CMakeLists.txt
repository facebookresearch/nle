cmake_minimum_required(VERSION 3.12)
file(STRINGS "version.txt" NLE_VERSION)
project(nle VERSION ${NLE_VERSION})
message(STATUS "Building nle backend version: ${NLE_VERSION}")

set(INSTALL_BASE "$ENV{HOME}")

set(HACKDIR ${INSTALL_BASE}/nledir)

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

set(DATADIR "${INSTALL_BASE}/nledir")
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

set(NLE_SRC ${nle_SOURCE_DIR}/src)
set(NLE_INC ${nle_SOURCE_DIR}/include)
set(NLE_DAT ${nle_SOURCE_DIR}/dat)
set(NLE_UTIL ${nle_SOURCE_DIR}/util)

set(NLE_SRC_GEN ${nle_BINARY_DIR}/src)
set(NLE_INC_GEN ${nle_BINARY_DIR}/include)
set(NLE_DAT_GEN ${nle_BINARY_DIR}/dat)
set(NLE_UTIL_GEN ${nle_BINARY_DIR}/util)

add_subdirectory(util)
add_subdirectory(dat)
add_subdirectory(win/rl)

add_executable(nlehack ${NLEHACK_SRC})
add_dependencies(nlehack makedefs_headers winrl)

target_include_directories(nlehack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                          ${NLE_INC_GEN})

find_library(ZMQ zmq)
find_library(CURSES ncurses)
target_link_libraries(nlehack PRIVATE ${ZMQ} ${CURSES} winrl)

install(TARGETS nlehack DESTINATION ${BINDIR})
install(DIRECTORY DESTINATION ${VARDIR})
install(DIRECTORY DESTINATION ${VARDIR}/save)
install(DIRECTORY DESTINATION ${INSTDIR})
