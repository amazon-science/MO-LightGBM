#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "scs::scsdir" for configuration "Release"
set_property(TARGET scs::scsdir APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(scs::scsdir PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libscsdir.so.2.1.4"
  IMPORTED_SONAME_RELEASE "libscsdir.so.2.1.4"
  )

list(APPEND _IMPORT_CHECK_TARGETS scs::scsdir )
list(APPEND _IMPORT_CHECK_FILES_FOR_scs::scsdir "${_IMPORT_PREFIX}/lib64/libscsdir.so.2.1.4" )

# Import target "scs::scsindir" for configuration "Release"
set_property(TARGET scs::scsindir APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(scs::scsindir PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libscsindir.so.2.1.4"
  IMPORTED_SONAME_RELEASE "libscsindir.so.2.1.4"
  )

list(APPEND _IMPORT_CHECK_TARGETS scs::scsindir )
list(APPEND _IMPORT_CHECK_FILES_FOR_scs::scsindir "${_IMPORT_PREFIX}/lib64/libscsindir.so.2.1.4" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
