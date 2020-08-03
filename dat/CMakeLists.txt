set(VARDATND) # NOTE: this is empty for us
set(VARDATD
    bogusmon
    data
    engrave
    epitaph
    oracles
    options
    quest.dat
    rumors)
set(VARDAT ${VARDATD} ${VARDATND})

set(DATHELP
    help
    hh
    cmdhelp
    keyhelp
    history
    opthelp
    wizhelp)

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

set(DATNODLB ${VARDATND} license symbols)

# TODO(NN): try to glob .lev automatically
#
# NOTE: Using:
#
# install(DIRECTORY ${NLE_DAT_GEN} DESTINATION ${INSTDIR} FILES_MATCHING PATTERN
# "*.lev")
#
# results in `$HACKDIR/dat/*.lev` due to CWD. I've tried to get it to work, but
# given up after a couple of hours.
set(LEVS_GEN
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

# NOTE: *.lev files will be added later
set(DATDLB ${DATHELP} dungeon tribute ${VARDATD})

set(DAT ${DATNODLB} ${DATDLB})

# NOTE: This is missing *.lev, which gets added later in the install decl
set(NLE_DAT_OUTPUT
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
    nhdat)

set(ALL_DAT_NOTGEN
    license
    symbols
    help
    hh
    cmdhelp
    history
    opthelp
    wizhelp)

# We transform these as we use them later
list(TRANSFORM QUEST_LEVELS PREPEND ${NLE_DAT}/)
list(TRANSFORM SPECIAL_LEVELS PREPEND ${NLE_DAT}/)
list(TRANSFORM DAT PREPEND ${NLE_DAT}/)
list(TRANSFORM LEVS_GEN PREPEND ${NLE_DAT_GEN}/)

set(NLE_DAT_GEN_OUTPUT
    ${NLE_DAT_OUTPUT}
    spec_levs
    quest_levs
    perm
    record
    logfile
    xlogfile
    sysconf)
list(TRANSFORM NLE_DAT_GEN_OUTPUT PREPEND ${NLE_DAT_GEN}/)

add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/bogusmon.txt ${NLE_DAT}/engrave.txt
          ${NLE_DAT}/epitaph.txt
  OUTPUT ${NLE_DAT}/bogusmon ${NLE_DAT}/epitaph ${NLE_DAT}/engrave
  COMMAND $<TARGET_FILE:makedefs> -s
  # HACK see ../util/CMakeLists.txt
  WORKING_DIRECTORY ${NLE_DAT})

add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/data.base
  OUTPUT ${NLE_DAT}/data
  COMMAND $<TARGET_FILE:makedefs> -d
  WORKING_DIRECTORY ${NLE_DAT})

add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/rumors.tru ${NLE_DAT}/rumors.fal
  OUTPUT ${NLE_DAT}/rumors
  COMMAND $<TARGET_FILE:makedefs> -r
  WORKING_DIRECTORY ${NLE_DAT})

add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/quest.txt
  OUTPUT ${NLE_DAT}/quest.dat
  COMMAND $<TARGET_FILE:makedefs> -q
  WORKING_DIRECTORY ${NLE_DAT})

add_custom_command(
  DEPENDS makedefs ${NLE_DAT}/oracles.txt
  OUTPUT ${NLE_DAT}/oracles
  COMMAND $<TARGET_FILE:makedefs> -h
  WORKING_DIRECTORY ${NLE_DAT})

# HACK: depends on date.h to avoid race wrt. the rm command
add_custom_command(
  DEPENDS makedefs ${NLE_INC_GEN}/date.h
  OUTPUT ${NLE_DAT}/options
  COMMAND $<TARGET_FILE:makedefs> -v
  COMMAND rm ${NLE_INC}/date.h
  WORKING_DIRECTORY ${NLE_DAT})

# HACK: depends on bogusmon to avoid race wrt. makedef's grep.tmp file.
add_custom_command(
  DEPENDS makedefs dgn_comp ${NLE_DAT}/dungeon.def ${NLE_DAT}/bogusmon
  OUTPUT ${NLE_DAT}/dungeon ${NLE_DAT}/dungeon.pdf
  COMMAND $<TARGET_FILE:makedefs> -e
  COMMAND $<TARGET_FILE:dgn_comp> dungeon.pdf
  WORKING_DIRECTORY ${NLE_DAT})

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

# NOTE: originally this only depended on options, however that's not quite
# right, as dlb needs to tar generated levels.
#
# We also assume that dat/options contains librarian (since we build with -DDLB)
add_custom_command(
  # originally this (technically) only depended on options
  DEPENDS dlb options ${DATDLB}
  OUTPUT ${NLE_DAT}/nhdat
  COMMAND LC_ALL=C $<TARGET_FILE:dlb> cf nhdat ${DATDLB}
  WORKING_DIRECTORY ${NLE_DAT})

set(NLE_DAT_OUTPUT_MOVED ${NLE_DAT_OUTPUT})
list(TRANSFORM NLE_DAT_OUTPUT_MOVED PREPEND ${NLE_DAT_GEN}/)
add_custom_command(
  DEPENDS ${NLE_DAT}/nhdat
  OUTPUT ${NLE_DAT_OUTPUT_MOVED}
  COMMAND mv ${NLE_DAT_OUTPUT} ${NLE_DAT_GEN}
  WORKING_DIRECTORY ${NLE_DAT})

add_custom_target(dat ALL DEPENDS ${NLE_DAT_GEN_OUTPUT} ${ALL_DAT_NOTGEN})

install(FILES ${NLE_DAT_GEN_OUTPUT} DESTINATION ${INSTDIR})
install(FILES ${ALL_DAT_NOTGEN} DESTINATION ${INSTDIR})

# NOTE: NetHack normally also generates a shell script at ~bin/nethack, however
# we likely do not need this, as most users currently have straightforward
# systems settings. It would be awesome though if someone were to run NLE on
# some old systems with weird pagers and stuff, at some point, eventually...
# after we solve AI :)
