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
file(MAKE_DIRECTORY ${NLE_DAT_GEN})

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
  COMMAND $<TARGET_FILE:makedefs> -v
  # HACK: due to makedefs being stubborn about in-source build
  COMMAND mv ${NLE_INC}/date.h ${NLE_INC_GEN}/date.h
  COMMAND rm ${NLE_DAT}/options # Since we don't need this here yet
  WORKING_DIRECTORY ../)

add_custom_command(
  DEPENDS makedefs
  OUTPUT ${NLE_INC_GEN}/onames.h
  COMMAND $<TARGET_FILE:makedefs> -o
  COMMAND mv ${NLE_INC}/onames.h ${NLE_INC_GEN}/onames.h
  WORKING_DIRECTORY ../)

add_custom_command(
  DEPENDS makedefs
  OUTPUT ${NLE_INC_GEN}/pm.h
  COMMAND $<TARGET_FILE:makedefs> -p
  COMMAND mv ${NLE_INC}/pm.h ${NLE_INC_GEN}/pm.h
  WORKING_DIRECTORY ../)

add_custom_command(
  OUTPUT ${NLE_UTIL_GEN}/dgn_parser.c ${NLE_UTIL_GEN}/dgn_comp.h
  COMMAND bison -y --defines=${NLE_UTIL_GEN}/dgn_comp.h -o
          ${NLE_UTIL_GEN}/dgn_parser.c ${NLE_UTIL}/dgn_comp.y
  DEPENDS ${NLE_UTIL}/dgn_comp.y)

add_custom_command(
  OUTPUT ${NLE_UTIL_GEN}/dgn_scanner.c
  COMMAND flex -o${NLE_UTIL_GEN}/dgn_scanner.c ${NLE_UTIL}/dgn_comp.l
  DEPENDS ${NLE_UTIL}/dgn_comp.l)

add_custom_command(
  OUTPUT ${NLE_UTIL_GEN}/lev_parser.c ${NLE_UTIL_GEN}/lev_comp.h
  COMMAND bison -y --defines=${NLE_UTIL_GEN}/lev_comp.h -o
          ${NLE_UTIL_GEN}/lev_parser.c ${NLE_UTIL}/lev_comp.y
  DEPENDS ${NLE_UTIL}/lev_comp.y)

add_custom_command(
  OUTPUT ${NLE_UTIL_GEN}/lev_scanner.c
  COMMAND flex -o${NLE_UTIL_GEN}/lev_scanner.c ${NLE_UTIL}/lev_comp.l
  DEPENDS ${NLE_UTIL}/lev_comp.l)

add_custom_target(makedefs_headers DEPENDS ${MAKEDEFS_HEADERS})
add_dependencies(lev_comp makedefs_headers)
add_dependencies(dgn_comp makedefs_headers)

install(FILES ${NLE_UTIL_GEN}/recover DESTINATION ${INSTDIR})
