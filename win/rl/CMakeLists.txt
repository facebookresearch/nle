# TODO(NN): we should rename this to winnle
add_library(winrl winrl.cc)
set_target_properties(winrl PROPERTIES CXX_STANDARD 14)

add_custom_target(
  message_generated COMMAND flatc -o ${NLE_INC_GEN} --cpp
                            ${nle_SOURCE_DIR}/win/rl/message.fbs)

target_include_directories(
  winrl
  PRIVATE ${NLE_INC_GEN} # message_generated.h
          ${NLE_INC} # hack.h
          /usr/local/include) # flatbuffers.h

find_library(FLATBUFFERS flatbuffers)
target_link_libraries(winrl PRIVATE ${FLATBUFFERS})
add_dependencies(winrl message_generated)

# NOTE: Below here should all be optional (but required for the python
# interface)

add_library(helper EXCLUDE_FROM_ALL helper.cc)
set_target_properties(helper PROPERTIES CXX_STANDARD 14)

# TODO(NN): We should probably be more defensive here (and check
# Python_Interpreter_FOUND)
find_package(Python COMPONENTS Interpreter Development)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import pybind11; print(pybind11.get_include(), end='')"
  OUTPUT_VARIABLE PYBIND11_INCLUDE_DIRS)
message(STATUS "Pybind include dirs: ${PYBIND11_INCLUDE_DIRS}")

target_include_directories(
  helper PRIVATE ${NLE_INC} ${NLE_INC_GEN} ${PYBIND11_INCLUDE_DIRS}
                 ${Python_INCLUDE_DIRS})

# NOTE: THis makes this build not quite out-of-source :(
add_custom_target(
  message_generated_py
  DEPENDS helper
  COMMAND flatc -o ${nle_SOURCE_DIR}/nle/fbs --python
          ${nle_SOURCE_DIR}/win/rl/message.fbs)

add_custom_target(fbs DEPENDS helper)
