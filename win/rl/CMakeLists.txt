# TODO(NN): we should rename this to winnle
add_library(winrl winrl.cc)
set_target_properties(winrl PROPERTIES CXX_STANDARD 14)
add_dependencies(winrl message_generated nledat)

add_custom_target(
  message_generated COMMAND flatc -o ${NLE_INC_GEN} --cpp
                            ${nle_SOURCE_DIR}/win/rl/message.fbs)

find_path(ZMQ_INCLUDE_DIR NAMES zmq.hpp)
find_library(ZMQ_LIBRARIES NAMES zmq)
message(STATUS "ZMQ include: ${ZMQ_INCLUDE_DIR}")
message(STATUS "ZMQ libs: ${ZMQ_LIBRARIES}")

find_path(FLATBUFFERS_INCLUDE_DIR NAMES flatbuffers/flatbuffers.h)
find_library(FLATBUFFERS_LIBRARIES NAMES flatbuffers)
message(STATUS "Flatbuffers include: ${FLATBUFFERS_INCLUDE_DIR}")
message(STATUS "Flatbuffers libs: ${FLATBUFFERS_LIBRARIES}")

target_include_directories(
  winrl
  PRIVATE ${NLE_INC_GEN} # message_generated.h
          ${NLE_INC} # hack.h
          ${ZMQ_INCLUDE_DIR} ${FLATBUFFERS_INCLUDE_DIR})

target_link_libraries(winrl PRIVATE ${FLATBUFFERS_LIBRARIES} ${ZMQ_LIBRARIES})

# NOTE: Below here should all be optional (but required for the python
# interface)

add_subdirectory(${nle_SOURCE_DIR}/third_party/pybind11 pybind11)

pybind11_add_module(helper EXCLUDE_FROM_ALL helper.cc)
set_target_properties(helper PROPERTIES CXX_STANDARD 14)
target_include_directories(helper PRIVATE ${NLE_INC} ${NLE_INC_GEN})

# These are required to be compiled statically for helper
add_library(monst STATIC ${NLE_SRC}/monst.c)
add_library(decl STATIC ${NLE_SRC}/decl.c)
add_library(drawing STATIC ${NLE_SRC}/drawing.c)

add_dependencies(monst nleutil)
add_dependencies(decl nleutil)
add_dependencies(drawing nleutil)

target_include_directories(monst PRIVATE ${NLE_INC} ${NLE_INC_GEN})
target_include_directories(decl PRIVATE ${NLE_INC} ${NLE_INC_GEN})
target_include_directories(drawing PRIVATE ${NLE_INC} ${NLE_INC_GEN})

set_target_properties(monst PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(decl PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(drawing PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(helper PRIVATE monst decl drawing)

# NOTE: This makes this build not quite out-of-source.
add_custom_target(
  message_generated_py
  COMMAND flatc -o ${nle_SOURCE_DIR}/win/rl --python
          ${nle_SOURCE_DIR}/win/rl/message.fbs
  COMMAND cp -rf ${nle_SOURCE_DIR}/win/rl/nle/fbs ${PYTHON_SRC_PARENT}/nle)

add_custom_target(fbs DEPENDS helper message_generated_py)
set_target_properties(fbs PROPERTIES EXCLUDE_FROM_ALL True)

# NOTE: Also dirtying source here.
install(
  TARGETS helper
  RUNTIME DESTINATION ${PYTHON_SRC_PARENT}/nle/nethack
  ARCHIVE DESTINATION ${PYTHON_SRC_PARENT}/nle/nethack
  LIBRARY DESTINATION ${PYTHON_SRC_PARENT}/nle/nethack OPTIONAL)
