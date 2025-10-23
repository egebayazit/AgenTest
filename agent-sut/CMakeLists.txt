cmake_minimum_required(VERSION 3.20)
project(agent_sut LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DUNICODE -D_UNICODE -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_WINSOCKAPI_ -D_WIN32_WINNT=0x0A00)

if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/action
  ${CMAKE_CURRENT_SOURCE_DIR}/win
  ${CMAKE_CURRENT_SOURCE_DIR}/jvm
)

if (WIN32)
  find_program(MAVEN_EXECUTABLE NAMES mvn.cmd mvn)
else()
  find_program(MAVEN_EXECUTABLE NAMES mvn)
endif()
if(NOT MAVEN_EXECUTABLE)
  message(FATAL_ERROR "mvn executable not found in PATH. Please install Maven.")
endif()
if (WIN32)
  get_filename_component(_MAVEN_EXT "${MAVEN_EXECUTABLE}" EXT)
  if (_MAVEN_EXT STREQUAL "")
    set(_MAVEN_CMD "${MAVEN_EXECUTABLE}.cmd")
    if (EXISTS "${_MAVEN_CMD}")
      set(MAVEN_EXECUTABLE "${_MAVEN_CMD}" CACHE FILEPATH "" FORCE)
    endif()
  endif()
endif()

find_program(JLINK_EXECUTABLE NAMES jlink PATHS "$ENV{JAVA_HOME}/bin" DOC "Path to jlink tool")
if(NOT JLINK_EXECUTABLE)
  message(FATAL_ERROR "jlink executable not found. Please install JDK 21+ and ensure JAVA_HOME is set.")
endif()

set(JVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/jvm")
set(JVM_AGENT_JAR "${JVM_DIR}/target/jvm-element-finder-1.0-SNAPSHOT-jar-with-dependencies.jar")
set(JVM_RUNTIME_DIR "${JVM_DIR}/runtime")
set(JVM_RUNTIME_ZIP "${JVM_DIR}/runtime.zip")
set(JVM_AGENT_RC "${CMAKE_BINARY_DIR}/jvm_agent.rc")

add_custom_command(
  OUTPUT "${JVM_AGENT_JAR}"
  COMMAND ${CMAKE_COMMAND} -E chdir "${JVM_DIR}" "${MAVEN_EXECUTABLE}" -q -DskipTests package
  BYPRODUCTS "${JVM_AGENT_JAR}"
  COMMENT "Building JVM agent jar"
  VERBATIM
)

add_custom_target(jvm_agent_jar ALL DEPENDS "${JVM_AGENT_JAR}")

add_custom_command(
  OUTPUT "${JVM_RUNTIME_ZIP}"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${JVM_RUNTIME_DIR}" "${JVM_RUNTIME_ZIP}"
  COMMAND "${JLINK_EXECUTABLE}" --add-modules java.base,java.instrument,java.management,java.desktop,java.logging,java.datatransfer,java.naming,jdk.attach,jdk.management,jdk.unsupported --strip-java-debug-attributes --no-header-files --no-man-pages --output "${JVM_RUNTIME_DIR}"
  COMMAND ${CMAKE_COMMAND} -E tar cf "${JVM_RUNTIME_ZIP}" --format=zip -- "${JVM_RUNTIME_DIR}"
  BYPRODUCTS "${JVM_RUNTIME_DIR}"
  COMMENT "Building JVM runtime image"
  VERBATIM
)

add_custom_target(jvm_runtime_zip ALL DEPENDS "${JVM_RUNTIME_ZIP}")

file(TO_NATIVE_PATH "${JVM_AGENT_JAR}" JVM_JAR_NATIVE_PATH)
string(REPLACE "\\" "\\\\" JVM_JAR_RC_PATH "${JVM_JAR_NATIVE_PATH}")
file(TO_NATIVE_PATH "${JVM_RUNTIME_ZIP}" JVM_RUNTIME_NATIVE_PATH)
string(REPLACE "\\" "\\\\" JVM_RUNTIME_RC_PATH "${JVM_RUNTIME_NATIVE_PATH}")
configure_file("${JVM_DIR}/jvm_agent.rc.in" "${JVM_AGENT_RC}" @ONLY)

add_executable(agent_sut
  server_main.cpp
  action/action_handler.cpp
  action/action_robot.cpp
  win/uia_utils.cpp
  win/capture.cpp
  win/base64.cpp
  jvm/jvm_bridge.cpp
)

target_sources(agent_sut PRIVATE "${JVM_AGENT_RC}")
add_dependencies(agent_sut jvm_agent_jar jvm_runtime_zip)

target_link_libraries(agent_sut
  PRIVATE
    user32
    Uiautomationcore
    Ole32 OleAut32
    Gdi32
    Windowscodecs
    Dwmapi
    Ws2_32
    Shcore
)

set_target_properties(agent_sut PROPERTIES OUTPUT_NAME "agent_sut")

# Usage instructions
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build --config Release
