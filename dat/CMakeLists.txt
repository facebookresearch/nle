set(SPECIAL_LEVELS
    bigroom.des
    castle.des
    endgame.des
    gehennom.des
    knox.des
    medusa.des
    mines.des
    oracle.des
    sokoban.des
    tower.des
    yendor.des)
list(TRANSFORM SPECIAL_LEVELS PREPEND ${NLE_DAT}/)

set(QUEST_LEVELS
    Arch.des
    Barb.des
    Caveman.des
    Healer.des
    Knight.des
    Monk.des
    Priest.des
    Ranger.des
    Rogue.des
    Samurai.des
    Tourist.des
    Valkyrie.des
    Wizard.des)

list(TRANSFORM QUEST_LEVELS PREPEND ${NLE_DAT}/)

# configure_file(${NLE_DAT}/history history COPYONLY)

# makedefs -s bogusmon epitaph engrave
add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/bogusmon.txt ${NLE_DAT}/engrave.txt
          ${NLE_DAT}/epitaph.txt
  OUTPUT bogusmon epitaph engrave
  COMMAND $<TARGET_FILE:makedefs> ARGS -s
  WORKING_DIRECTORY ../)

# makedefs -d data.base
add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/data.base
  OUTPUT data
  COMMAND $<TARGET_FILE:makedefs> -d # ${NLE_DAT}/data.base data
  WORKING_DIRECTORY ../) # HACK see ../util/CMakeLists.txt

# makedefs -r rumors.tru rumors.fal
add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/rumors.tru ${NLE_DAT}/rumors.fal
  OUTPUT rumors
  COMMAND $<TARGET_FILE:makedefs> -r # ${NLE_DAT}/rumors.tru
                                     # ${NLE_DAT}/rumors.fal rumors
  WORKING_DIRECTORY ../)

# makedefs -q quest.txt
add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/quest.txt
  OUTPUT quest.dat
  COMMAND $<TARGET_FILE:makedefs> -q # ${NLE_DAT}/quest.txt quest.dat
  WORKING_DIRECTORY ../)

# makedefs -h oracles
add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/oracles.txt
  OUTPUT oracles
  COMMAND $<TARGET_FILE:makedefs> -h # ${NLE_DAT}/oracles.txt oracles
  WORKING_DIRECTORY ../)

# makedefs -v options NOTE: Should already have been made when include/date.h
# was created
add_custom_command(
  DEPENDS makedefs
  OUTPUT options
  COMMAND $<TARGET_FILE:makedefs> -v
  WORKING_DIRECTORY ../)

# makedefs -e dungeon
add_custom_command(
  DEPENDS makedefs dgn_comp ${NLE_DAT}/dungeon.def
  OUTPUT dungeon.pdf
  COMMAND $<TARGET_FILE:makedefs> -e
  COMMAND $<TARGET_FILE:dgn_comp> ${NLE_DAT}/dungeon.pdf
  WORKING_DIRECTORY ../)

# spec_levs
add_custom_command(
  DEPENDS lev_comp
  OUTPUT spec_levs
  COMMAND $<TARGET_FILE:lev_comp> ${SPECIAL_LEVELS}
  COMMAND ${CMAKE_COMMAND} -E touch spec_levs)

# quest_levs
add_custom_command(
  DEPENDS lev_comp
  OUTPUT quest_levs
  COMMAND $<TARGET_FILE:lev_comp> ${QUEST_LEVELS}
  COMMAND ${CMAKE_COMMAND} -E touch quest_levs)

# pseudo-target: depend on all other generated data
add_custom_target(
  nle_data_files ALL
  DEPENDS bogusmon
          epitaph
          engrave
          data
          rumors
          quest.dat
          oracles
          options
          dungeon.pdf
          spec_levs
          quest_levs)

install(FILES ${NLE_DAT_GEN}/nhdat DESTINATION ${DATADIR})
install(FILES license DESTINATION ${DATADIR})
