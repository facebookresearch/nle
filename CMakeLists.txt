cmake_minimum_required(VERSION 3.10)
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

add_executable(nlehack ${nle_SRC})

set_target_properties(nlehack PROPERTIES CXX_STANDARD 14)

# For now we need to run makedef first
target_include_directories(nlehack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                          /usr/local/include)

# target_link_libraries(nle PUBLIC nethack)
target_link_directories(nlehack PUBLIC /usr/local/lib)
target_link_libraries(nlehack PUBLIC ncurses m zmq lua)
target_link_libraries(nlehack PRIVATE flatbuffers)
