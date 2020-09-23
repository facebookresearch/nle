find_program(FLEX_EXECUTABLE NAMES flex flex.exe)
find_program(BISON_EXECUTABLE NAMES bison bison.exe)

set(MAKEDEFS_SRC makedefs.c ${NLE_SRC}/monst.c ${NLE_SRC}/objects.c)

set(DGN_COMP_SRC dgn_main.c panic.c ${NLE_SRC}/alloc.c
                 ${NLE_UTIL_GEN}/dgn_parser.c ${NLE_UTIL_GEN}/dgn_scanner.c)

set(LEV_COMP_SRC
    lev_main.c
    panic.c
    ${NLE_SRC}/decl.c
    ${NLE_SRC}/monst.c
    ${NLE_SRC}/objects.c
    ${NLE_SRC}/alloc.c
    ${NLE_SRC}/drawing.c
    ${NLE_UTIL_GEN}/lev_parser.c
    ${NLE_UTIL_GEN}/lev_scanner.c)

set(MAKEDEFS_HEADERS ${NLE_INC_GEN}/date.h ${NLE_INC_GEN}/onames.h
                     ${NLE_INC_GEN}/pm.h)

set(DLB_SRC dlb_main.c ${NLE_SRC}/dlb.c panic.c ${NLE_SRC}/alloc.c)

file(MAKE_DIRECTORY ${NLE_INC_GEN})

add_executable(makedefs ${MAKEDEFS_SRC})
target_include_directories(makedefs PRIVATE ${NLE_INC} ${NLE_INC_GEN}
                                            ${NLE_UTIL_GEN})
add_executable(dgn_comp ${DGN_COMP_SRC})
target_include_directories(dgn_comp PRIVATE ${NLE_INC} ${NLE_INC_GEN}
                                            ${NLE_UTIL_GEN})
add_executable(lev_comp ${LEV_COMP_SRC})
target_include_directories(lev_comp PRIVATE ${NLE_INC} ${NLE_INC_GEN}
                                            ${NLE_UTIL_GEN})
add_executable(dlb ${DLB_SRC})
target_include_directories(dlb PRIVATE ${NLE_INC} ${NLE_INC_GEN}
                                       ${NLE_UTIL_GEN})
add_executable(recover recover.c)
target_include_directories(recover PRIVATE ${NLE_INC} ${NLE_INC_GEN}
                                           ${NLE_UTIL_GEN})

add_custom_command(
  DEPENDS makedefs
  OUTPUT ${NLE_INC_GEN}/date.h
  COMMAND $<TARGET_FILE:makedefs> ARGS -v)

add_custom_command(
  DEPENDS makedefs
  OUTPUT ${NLE_INC_GEN}/onames.h
  COMMAND $<TARGET_FILE:makedefs> ARGS -o)

add_custom_command(
  DEPENDS makedefs
  OUTPUT ${NLE_INC_GEN}/pm.h
  COMMAND $<TARGET_FILE:makedefs> ARGS -p)

add_custom_command(
  OUTPUT ${NLE_UTIL_GEN}/dgn_parser.c ${NLE_UTIL_GEN}/dgn_comp.h
  COMMAND bison -y --defines=${NLE_UTIL_GEN}/dgn_comp.h -o
          ${NLE_UTIL_GEN}/dgn_parser.c ${NLE_UTIL}/dgn_comp.y
  DEPENDS ${NLE_UTIL}/dgn_comp.y)

add_custom_command(
  OUTPUT ${NLE_UTIL_GEN}/dgn_scanner.c
  COMMAND ${FLEX_EXECUTABLE} ARGS -o${NLE_UTIL_GEN}/dgn_scanner.c
          ${NLE_UTIL}/dgn_comp.l
  DEPENDS ${NLE_UTIL}/dgn_comp.l)

add_custom_command(
  OUTPUT ${NLE_UTIL_GEN}/lev_parser.c ${NLE_UTIL_GEN}/lev_comp.h
  COMMAND ${BISON_EXECUTABLE} ARGS -y --defines=${NLE_UTIL_GEN}/lev_comp.h -o
          ${NLE_UTIL_GEN}/lev_parser.c ${NLE_UTIL}/lev_comp.y
  DEPENDS ${NLE_UTIL}/lev_comp.y)

add_custom_command(
  OUTPUT ${NLE_UTIL_GEN}/lev_scanner.c
  COMMAND ${FLEX_EXECUTABLE} ARGS -o${NLE_UTIL_GEN}/lev_scanner.c
          ${NLE_UTIL}/lev_comp.l
  DEPENDS ${NLE_UTIL}/lev_comp.l)

add_custom_target(util DEPENDS ${MAKEDEFS_HEADERS} recover)

# NOTE: util is dependent on these two
add_dependencies(lev_comp util)
add_dependencies(dgn_comp util)

set(MAN_PAGES ${NLE_DOC}/makedefs.6 ${NLE_DOC}/dgn_comp.6 ${NLE_DOC}/lev_comp.6
              ${NLE_DOC}/dlb.6 ${NLE_DOC}/recover.6 ${NLE_DOC}/nethack.6)

set(TXT_DOCS
    ${NLE_DOC}/makedefs.txt ${NLE_DOC}/dgn_comp.txt ${NLE_DOC}/lev_comp.txt
    ${NLE_DOC}/dlb.txt ${NLE_DOC}/recover.txt ${NLE_DOC}/nethack.txt)

install(TARGETS makedefs dgn_comp lev_comp dlb recover DESTINATION ${INSTDIR})
install(FILES ${MAN_PAGES} DESTINATION ${INSTDIR}/man)
install(FILES ${TXT_DOCS} DESTINATION ${INSTDIR}/doc)
