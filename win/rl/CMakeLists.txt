# TODO(NN): we should rename this to winnle
add_library(winrl winrl.cc)

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
set_target_properties(winrl PROPERTIES CXX_STANDARD 14)
add_dependencies(winrl message_generated)
