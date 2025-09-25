cmake_minimum_required(VERSION 3.20)
project(agent_sut LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows header ve winsock çakışmalarını önle
add_definitions(-DUNICODE -D_UNICODE -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_WINSOCKAPI_)

# MSVC'de VC++ runtime'ı statik linkle (vcruntime DLL'ine ihtiyaç kalmaz)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_executable(agent_sut
  src/main.cpp
  src/uia_utils.cpp
  src/capture.cpp
)

target_include_directories(agent_sut PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agent_sut
  Uiautomationcore Ole32 OleAut32 User32 Gdi32 Windowscodecs Dwmapi Ws2_32
)
