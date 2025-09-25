cmake_minimum_required(VERSION 3.20)
project(sut_action LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_library(sut_action STATIC
    action_robot.cpp
    action_handler.cpp
)
target_link_libraries(sut_action PRIVATE user32)

# ---- opsiyonel: action tarafını tek başına koşturmak için küçük HTTP server
option(BUILD_SUT_ACTION_SERVER "Build standalone /action server from action folder" ON)
if(BUILD_SUT_ACTION_SERVER)
  add_executable(sut_action_server main_action.cpp)
  target_link_libraries(sut_action_server PRIVATE sut_action)
endif()

##Derleme (yalnız action klasörü):
#cd agent-sut/action
#cmake -S . -B build
#cmake --build build --config Release