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

add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/bogusmon.txt ${NLE_DAT}/engrave.txt
          ${NLE_DAT}/epitaph.txt
  OUTPUT bogusmon epitaph engrave
  COMMAND $<TARGET_FILE:makedefs> ARGS -s
  COMMAND mv ${NLE_DAT}/bogusmon ${NLE_DAT}/engrave ${NLE_DAT}/epitaph
          ${NLE_DAT_GEN}/
  WORKING_DIRECTORY ../)

add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/data.base
  OUTPUT data
  COMMAND $<TARGET_FILE:makedefs> -d
  # HACK see ../util/CMakeLists.txt
  COMMAND mv ${NLE_DAT}/data ${NLE_DAT_GEN}/
  WORKING_DIRECTORY ../)

add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/rumors.tru ${NLE_DAT}/rumors.fal
  OUTPUT rumors
  COMMAND $<TARGET_FILE:makedefs> -r
  COMMAND mv ${NLE_DAT}/rumors ${NLE_DAT_GEN}/
  WORKING_DIRECTORY ../)

add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/quest.txt
  OUTPUT quest.dat
  COMMAND $<TARGET_FILE:makedefs> -q
  COMMAND mv ${NLE_DAT}/quest.dat ${NLE_DAT_GEN}/
  WORKING_DIRECTORY ../)

add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/oracles.txt
  OUTPUT oracles
  COMMAND $<TARGET_FILE:makedefs> -h
  COMMAND mv ${NLE_DAT}/oracles ${NLE_DAT_GEN}/
  WORKING_DIRECTORY ../)

add_custom_command(
  DEPENDS makedefs
  OUTPUT options
  COMMAND $<TARGET_FILE:makedefs> -v
  COMMAND mv ${NLE_DAT}/options ${NLE_DAT_GEN}/
  COMMAND rm ${NLE_INC}/date.h # We don't need this here
  WORKING_DIRECTORY ../)

add_custom_command(
  DEPENDS makedefs dgn_comp ${NLE_DAT}/dungeon.def
  OUTPUT dungeon.pdf dungeon
  COMMAND $<TARGET_FILE:makedefs> -e
  COMMAND $<TARGET_FILE:dgn_comp> ${NLE_DAT}/dungeon.pdf
  COMMAND mv ${NLE_DAT}/dungeon.pdf ${NLE_DAT}/dungeon ${NLE_DAT_GEN}/
  WORKING_DIRECTORY ../)

add_custom_command(
  DEPENDS lev_comp
  OUTPUT spec_levs
  COMMAND $<TARGET_FILE:lev_comp> ${SPECIAL_LEVELS}
  COMMAND ${CMAKE_COMMAND} -E touch spec_levs)

add_custom_command(
  DEPENDS lev_comp
  OUTPUT quest_levs
  COMMAND $<TARGET_FILE:lev_comp> ${QUEST_LEVELS}
  COMMAND ${CMAKE_COMMAND} -E touch quest_levs)

add_custom_command(
  OUTPUT perm record logfile xlogfile sysconf
  COMMAND ${CMAKE_COMMAND} -E touch perm record logfile xlogfile sysconf)

# NOTE: NetHack normally also generates a shell script at ~bin/nethack, however
# we likely do not need this, as most users currently have straightforward
# systems settings. It would be awesome though if someone were to run NLE on
# some old systems with weird pagers and stuff, at some point, eventually...
# after we solve AI :)

# NOTE: This is missing *.lev, which gets added later in the install decl
set(ALL_DAT_OUTPUT
    bogusmon
    epitaph
    engrave
    data
    rumors
    quest.dat
    oracles
    options
    dungeon
    dungeon.pdf
    spec_levs
    quest_levs
    perm
    record
    logfile
    xlogfile
    sysconf)

set(ALL_DAT_GEN_OUTPUT ${ALL_DAT_OUTPUT})
list(TRANSFORM ALL_DAT_GEN_OUTPUT PREPEND ${NLE_DAT_GEN}/)

set(ALL_DAT_NOTGEN
    license
    symbols
    help
    hh
    cmdhelp
    history
    opthelp
    wizhelp)

# TODO(NN): try to glob .lev automatically
#
# NOTE: Using:
#
# install(DIRECTORY ${NLE_DAT_GEN} DESTINATION ${INSTDIR} FILES_MATCHING PATTERN
# "*.lev")
#
# results in `$HACKDIR/dat/*.lev` due to CWD. I've tried to get it to work, but
# given up after a couple of hours :(
set(ALL_DAT_GEN_LEV
    Arc-fila.lev
    Arc-filb.lev
    Arc-goal.lev
    Arc-loca.lev
    Arc-strt.lev
    Bar-fila.lev
    Bar-filb.lev
    Bar-goal.lev
    Bar-loca.lev
    Bar-strt.lev
    Cav-fila.lev
    Cav-filb.lev
    Cav-goal.lev
    Cav-loca.lev
    Cav-strt.lev
    Hea-fila.lev
    Hea-filb.lev
    Hea-goal.lev
    Hea-loca.lev
    Hea-strt.lev
    Kni-fila.lev
    Kni-filb.lev
    Kni-goal.lev
    Kni-loca.lev
    Kni-strt.lev
    Mon-fila.lev
    Mon-filb.lev
    Mon-goal.lev
    Mon-loca.lev
    Mon-strt.lev
    Pri-fila.lev
    Pri-filb.lev
    Pri-goal.lev
    Pri-loca.lev
    Pri-strt.lev
    Ran-fila.lev
    Ran-filb.lev
    Ran-goal.lev
    Ran-loca.lev
    Ran-strt.lev
    Rog-fila.lev
    Rog-filb.lev
    Rog-goal.lev
    Rog-loca.lev
    Rog-strt.lev
    Sam-fila.lev
    Sam-filb.lev
    Sam-goal.lev
    Sam-loca.lev
    Sam-strt.lev
    Tou-fila.lev
    Tou-filb.lev
    Tou-goal.lev
    Tou-loca.lev
    Tou-strt.lev
    Val-fila.lev
    Val-filb.lev
    Val-goal.lev
    Val-loca.lev
    Val-strt.lev
    Wiz-fila.lev
    Wiz-filb.lev
    Wiz-goal.lev
    Wiz-loca.lev
    Wiz-strt.lev
    air.lev
    asmodeus.lev
    astral.lev
    baalz.lev
    bigrm-1.lev
    bigrm-10.lev
    bigrm-2.lev
    bigrm-3.lev
    bigrm-4.lev
    bigrm-5.lev
    bigrm-6.lev
    bigrm-7.lev
    bigrm-8.lev
    bigrm-9.lev
    castle.lev
    earth.lev
    fakewiz1.lev
    fakewiz2.lev
    fire.lev
    juiblex.lev
    knox.lev
    medusa-1.lev
    medusa-2.lev
    medusa-3.lev
    medusa-4.lev
    minefill.lev
    minend-1.lev
    minend-2.lev
    minend-3.lev
    minetn-1.lev
    minetn-2.lev
    minetn-3.lev
    minetn-4.lev
    minetn-5.lev
    minetn-6.lev
    minetn-7.lev
    oracle.lev
    orcus.lev
    sanctum.lev
    soko1-1.lev
    soko1-2.lev
    soko2-1.lev
    soko2-2.lev
    soko3-1.lev
    soko3-2.lev
    soko4-1.lev
    soko4-2.lev
    tower1.lev
    tower2.lev
    tower3.lev
    valley.lev
    water.lev
    wizard1.lev
    wizard2.lev
    wizard3.lev)
list(TRANSFORM ALL_DAT_GEN_LEV PREPEND ${NLE_DAT_GEN}/)

add_custom_target(nledat ALL DEPENDS ${ALL_DAT_OUTPUT} ${ALL_DAT_NOTGEN})

install(FILES ${ALL_DAT_GEN_OUTPUT} ${ALL_DAT_GEN_LEV} DESTINATION ${INSTDIR})

install(FILES ${ALL_DAT_NOTGEN} DESTINATION ${INSTDIR})
