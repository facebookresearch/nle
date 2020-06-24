cmake_minimum_required(VERSION 3.12)
file(STRINGS "version.txt" NLE_VERSION)
project(nle VERSION ${NLE_VERSION})
message(STATUS "Building nle backend version: ${NLE_VERSION}")

set(INSTALL_BASE "$ENV{HOME}")

add_compile_definitions(
  GCC_WARN
  NOCLIPPING
  NOMAIL
  NOTPARMDECL
  HACKDIR="${INSTALL_BASE}/nledir"
  DEFAULT_WINDOW_SYS="tty"
  DLB)

set(DATADIR "${INSTALL_BASE}/nledir")
file(
  GLOB
  nle_SRC
  "src/*.c"
  "sys/share/posixregex.c"
  "sys/share/ioctl.c"
  "sys/unix/unixunix.c"
  "sys/share/unixtty.c"
  "sys/unix/unixres.c"
  "sys/unix/unixmain.c"
  "win/tty/*.c"
  "win/rl/winrl.cc")

set(NLE_SRC ${nle_SOURCE_DIR}/src)
set(NLE_INC ${nle_SOURCE_DIR}/include)
set(NLE_DAT ${nle_SOURCE_DIR}/dat)
set(NLE_UTIL ${nle_SOURCE_DIR}/util)

# set(NLE_SRC_GEN ${nle_BINARY_DIR}/nlehack/src) set(NLE_INC_GEN
# ${nle_BINARY_DIR}/nlehack/include) set(NLE_DAT_GEN
# ${nle_BINARY_DIR}/nlehack/dat) set(NLE_UTIL_GEN
# ${nle_BINARY_DIR}/nlehack/util)

set(NLE_SRC_GEN ${nle_BINARY_DIR}/src)
set(NLE_INC_GEN ${nle_BINARY_DIR}/include)
set(NLE_DAT_GEN ${nle_BINARY_DIR}/dat)
set(NLE_UTIL_GEN ${nle_BINARY_DIR}/util)

add_subdirectory(util)
add_subdirectory(dat)
add_subdirectory(win/rl)

# add_subdirectory(src)

add_executable(nlehack ${nle_SRC})
add_dependencies(nlehack makedefs_headers nle_generated)

set_target_properties(nlehack PROPERTIES CXX_STANDARD 14)

# For now we need to run makedef first
target_include_directories(
  nlehack
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${NLE_INC_GEN}
         # ${NLE_UTIL_GEN}
         /usr/local/include)

# target_link_libraries(nle PUBLIC nethack)
target_link_directories(nlehack PUBLIC /usr/local/lib)
target_link_libraries(nlehack PUBLIC ncurses m zmq lua)
target_link_libraries(nlehack PRIVATE flatbuffers)

message(STATUS "######################### ${NLE_SRC_GEN}")
message(STATUS "######################### ${NLE_SRC}")
