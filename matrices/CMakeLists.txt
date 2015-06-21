
FILE(GLOB sources *.cc)

GET_FILENAME_COMPONENT(prefix ${CMAKE_CURRENT_SOURCE_DIR} NAME)

FOREACH(ccfile ${sources})
  GET_FILENAME_COMPONENT(file ${ccfile} NAME_WE)
  SET(target ${prefix}_${file})
  ADD_EXECUTABLE(${target} ${ccfile})
  SET_TARGET_PROPERTIES(${target} PROPERTIES OUTPUT_NAME ${file})
  DEAL_II_SETUP_TARGET(${target})
ENDFOREACH()
