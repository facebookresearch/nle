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

set(DATDLB
    ${DATHELP}
    dungeon
    tribute
    bogusmon
    data
    engrave
    epitaph
    oracles
    options
    quest.dat
    rumors)

file(MAKE_DIRECTORY ${NLE_DAT_GEN})

# Inputs to makedef and lev_comp (and transitive inputs to dgn_comp and dlb).
set(ALL_DAT_NOTGEN
    ${DATHELP}
    ${SPECIAL_LEVELS}
    ${QUEST_LEVELS}
    bogusmon.txt
    data.base
    dungeon.def
    engrave.txt
    epitaph.txt
    oracles.txt
    quest.txt
    rumors.fal
    rumors.tru
    tribute)

# makedef and friends like having their inputs in the same directory.
foreach(filename ${ALL_DAT_NOTGEN})
  configure_file(${NLE_DAT}/${filename} ${filename} COPYONLY)
endforeach(filename)

# Dat files that are not inputs to any tools.
set(ALL_DAT_NOTGEN ${ALL_DAT_NOTGEN} license symbols)

# Result of running lev_comp.
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

add_custom_command(
  DEPENDS $<TARGET_FILE:makedefs> bogusmon.txt engrave.txt epitaph.txt
  OUTPUT bogusmon epitaph engrave
  COMMAND makedefs ARGS -s)

add_custom_command(
  DEPENDS $<TARGET_FILE:makedefs> data.base
  OUTPUT data
  COMMAND makedefs ARGS -d)

add_custom_command(
  DEPENDS $<TARGET_FILE:makedefs> rumors.tru rumors.fal
  OUTPUT rumors
  COMMAND makedefs ARGS -r)

add_custom_command(
  DEPENDS $<TARGET_FILE:makedefs> quest.txt
  OUTPUT quest.dat
  COMMAND makedefs ARGS -q)

add_custom_command(
  DEPENDS $<TARGET_FILE:makedefs> oracles.txt
  OUTPUT oracles
  COMMAND makedefs ARGS -h)

# Hack: Depend on date.h to avoid race.
add_custom_command(
  DEPENDS $<TARGET_FILE:makedefs> ${NLE_INC_GEN}/date.h
  OUTPUT options
  COMMAND makedefs ARGS -v)

# Hack: Depend on bogusmon to avoid race wrt. makedef's grep.tmp file. Best to
# be defensive about the paths (matters if NetHack's make install was run).
add_custom_command(
  DEPENDS $<TARGET_FILE:makedefs> $<TARGET_FILE:dgn_comp>
          ${NLE_DAT_GEN}/dungeon.def ${NLE_DAT_GEN}/bogusmon
  OUTPUT dungeon dungeon.pdf
  COMMAND makedefs ARGS -e
  COMMAND dgn_comp ARGS dungeon.pdf)

add_custom_command(
  DEPENDS lev_comp
  OUTPUT ${LEVS_GEN}
  COMMAND lev_comp ${SPECIAL_LEVELS}
  COMMAND lev_comp ${QUEST_LEVELS})

add_custom_command(
  OUTPUT perm record logfile xlogfile COMMAND ${CMAKE_COMMAND} -E touch perm
                                              record logfile xlogfile)

# Defensive dependencies for when we also build NetHack with make/make install.
list(TRANSFORM DATDLB PREPEND ${NLE_DAT_GEN}/ OUTPUT_VARIABLE
                                              NLE_DAT_GEN_DATDLB)
list(TRANSFORM LEVS_GEN PREPEND ${NLE_DAT_GEN}/ OUTPUT_VARIABLE
                                                NLE_DAT_GEN_LEVS_GEN)

add_custom_command(
  DEPENDS dlb ${NLE_DAT_GEN_DATDLB} ${NLE_DAT_GEN_LEVS_GEN}
  OUTPUT nhdat
  COMMAND LC_ALL=C $<TARGET_FILE:dlb> ARGS cf nhdat ${DATDLB} ${LEVS_GEN})

set(ALL_DAT_GEN nhdat perm record logfile xlogfile)
list(TRANSFORM ALL_DAT_GEN PREPEND ${NLE_DAT_GEN}/)

add_custom_target(dat ALL DEPENDS ${ALL_DAT_GEN} ${ALL_DAT_NOTGEN})

install(FILES ${ALL_DAT_GEN} DESTINATION ${INSTDIR})
install(FILES ${ALL_DAT_NOTGEN} DESTINATION ${INSTDIR}/dat)
install(DIRECTORY DESTINATION ${INSTDIR}/save)
