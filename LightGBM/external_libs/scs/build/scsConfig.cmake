set(scs_VERSION 2.1.4)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was scsConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################



if(NOT TARGET scs::scsdir)
  include("${CMAKE_CURRENT_LIST_DIR}/scsTargets.cmake")
endif()

# Compatibility
get_property(scs_scsdir_INCLUDE_DIR TARGET scs::scsdir PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
get_property(scs_scsindir_INCLUDE_DIR TARGET scs::scsindir PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

set(scs_LIBRARIES scs::scsdir scs::scsindir)
set(scs_INCLUDE_DIRS "${scs_scsdir_INCLUDE_DIR}" "${scs_scsindir_INCLUDE_DIR}")
list(REMOVE_DUPLICATES scs_INCLUDE_DIRS)


