#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ecos::ecos" for configuration "Debug"
set_property(TARGET ecos::ecos APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ecos::ecos PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libecos.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libecos.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS ecos::ecos )
list(APPEND _IMPORT_CHECK_FILES_FOR_ecos::ecos "${_IMPORT_PREFIX}/lib/libecos.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
