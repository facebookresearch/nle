set(MAKEDEFS_SRC makedefs.c ${NLE_SRC}/monst.c ${NLE_SRC}/objects.c)

set(DGN_COMP_SRC dgn_main.c panic.c ${NLE_UTIL_GEN}/dgn_parser.c
                 ${NLE_UTIL_GEN}/dgn_scanner.c)

set(LEV_COMP_SRC
    lev_main.c
    panic.c
    # ${NLE_SRC}/symclass.c
    ${NLE_SRC}/decl.c
    ${NLE_SRC}/monst.c
    ${NLE_SRC}/objects.c
    ${NLE_UTIL_GEN}/lev_parser.c
    ${NLE_UTIL_GEN}/lev_scanner.c)
set(MAKEDEFS_HEADERS ${NLE_INC_GEN}/date.h ${NLE_INC_GEN}/onames.h
                     ${NLE_INC_GEN}/pm.h)
set(DLB_SRC dlb_main.c ${NLE_SRC}/dlb.c)

file(MAKE_DIRECTORY ${NLE_INC_GEN})
file(MAKE_DIRECTORY ${NLE_DAT_GEN})

include_directories(${nle_SOURCE_DIR}/include ${nle_SOURCE_DIR}/nle/include
                    ${NLE_INC_GEN} ${NLE_UTIL_GEN})

# targets
add_executable(makedefs ${MAKEDEFS_SRC})
add_executable(dgn_comp ${DGN_COMP_SRC})
add_executable(lev_comp ${LEV_COMP_SRC})
add_executable(dlb ${DLB_SRC})

# get_property( MAKEDEFS_BIN TARGET makedefs PROPERTY LOCATION)

# create the headers named in MAKEDEFS_HEADERS
add_custom_command(
  OUTPUT ${NLE_INC_GEN}/date.h
  COMMAND $<TARGET_FILE:makedefs> -v # ${NLE_INC_GEN}/date.h
  COMMAND mv ${NLE_INC}/date.h ${NLE_INC_GEN}/date.h
  WORKING_DIRECTORY ../
  DEPENDS makedefs)

add_custom_command(
  OUTPUT ${NLE_INC_GEN}/onames.h
  COMMAND $<TARGET_FILE:makedefs> -o # ${NLE_INC_GEN}/onames.h
  COMMAND mv ${NLE_INC}/onames.h ${NLE_INC_GEN}/onames.h
  WORKING_DIRECTORY ../
  DEPENDS makedefs)

add_custom_command(
  OUTPUT ${NLE_INC_GEN}/pm.h
  COMMAND $<TARGET_FILE:makedefs> -p # ${NLE_INC_GEN}/pm.h
  COMMAND mv ${NLE_INC}/pm.h ${NLE_INC_GEN}/pm.h
  WORKING_DIRECTORY ../
  DEPENDS makedefs)

# generate parser & scanner for dgn_comp
add_custom_command(
  OUTPUT ${NLE_UTIL_GEN}/dgn_parser.c ${NLE_UTIL_GEN}/dgn_comp.h
  COMMAND bison -y --defines=${NLE_UTIL_GEN}/dgn_comp.h -o
          ${NLE_UTIL_GEN}/dgn_parser.c ${NLE_UTIL}/dgn_comp.y
  DEPENDS ${NLE_UTIL}/dgn_comp.y)

add_custom_command(
  OUTPUT ${NLE_UTIL_GEN}/dgn_scanner.c
  COMMAND flex -o${NLE_UTIL_GEN}/dgn_scanner.c ${NLE_UTIL}/dgn_comp.l
  DEPENDS ${NLE_UTIL}/dgn_comp.l)

# generate parser & scanner for lev_comp
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
add_dependencies(dgn_comp makedefs_headers)
add_dependencies(lev_comp makedefs_headers)
