add_custom_target(nle_generated COMMAND flatc -o ${NLE_INC_GEN} --cpp
                                        ${nle_SOURCE_DIR}/win/rl/message.fbs)
