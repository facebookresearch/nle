cmake_minimum_required(VERSION 3.15)
project(NetHack VERSION 3.6)

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
  AVOID_WIN_IOCTL
  SCORE_ON_BOTL
  # ANSI_DEFAULT NO_SIGNAL
  HACKDIR="${HACKDIR}"
  DEFAULT_WINDOW_SYS="tty"
  DLB)

set(NLE_SRC ${nle_SOURCE_DIR}/src)
set(NLE_INC ${nle_SOURCE_DIR}/include)
set(NLE_DAT ${nle_SOURCE_DIR}/dat)
set(NLE_UTIL ${nle_SOURCE_DIR}/util)
set(NLE_DOC ${nle_SOURCE_DIR}/doc)

set(NLE_SRC_GEN ${nle_BINARY_DIR}/src)
set(NLE_INC_GEN ${nle_BINARY_DIR}/include)
set(NLE_DAT_GEN ${nle_BINARY_DIR}/dat)
set(NLE_UTIL_GEN ${nle_BINARY_DIR}/util)

# EXCLUDE_FROM_ALL: Don't install this static library into /usr/local.
# add_subdirectory(third_party/deboost.context EXCLUDE_FROM_ALL)
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
  "sys/share/unixtty.c"
  "win/tty/*.c")

if(1)
  # nethack executable.
  add_executable(nethack ${NETHACK_SRC})
  target_link_libraries(nethack ncurses)
else()
  # libnethack.so ELF library.
  add_library(nethack OBJECT ${NETHACK_SRC})
  target_compile_options(nethack PUBLIC -target x86_64-apple-none-elf)

  add_custom_command(
    DEPENDS $<TARGET_OBJECTS:nethack>
    OUTPUT libnethack.so
    COMMAND x86_64-elf-ld -shared -o libnethack.so $<TARGET_OBJECTS:nethack>
    COMMAND_EXPAND_LISTS)

  add_custom_target(link-nethack ALL DEPENDS libnethack.so)
endif()

add_dependencies(nethack util dat)

target_include_directories(
  nethack
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${NLE_INC_GEN} /usr/local/include
         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtmt
         ${CMAKE_CURRENT_SOURCE_DIR}/third_party/deboost.context/include)
