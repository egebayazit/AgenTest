cmake_minimum_required(VERSION 3.20)
project(agent_sut LANGUAGES CXX)

# --- Dil standardı
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Ortak derleme tanımları
add_definitions(-DUNICODE -D_UNICODE -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_WINSOCKAPI_ -D_WIN32_WINNT=0x0A00)

# MSVC: statik runtime
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# --- Include path'ler
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}          # <-- kök (third_party/json.hpp için)
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/action
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# --- Tek executable
add_executable(agent_sut
  server_main.cpp

  # Action modülü (main_action.cpp dahil değil)
  action/action_handler.cpp
  action/action_robot.cpp

  # State modülü (src/main.cpp dahil değil)
  src/uia_utils.cpp
  src/capture.cpp
  src/base64.cpp
)

# --- Link kütüphaneleri
target_link_libraries(agent_sut
  PRIVATE
    user32
    Uiautomationcore
    Ole32 OleAut32
    Gdi32
    Windowscodecs
    Dwmapi
    Ws2_32
    Shcore         # <-- DPI awareness (SetProcessDpiAwareness / Context)
)

# Çıktı adı
set_target_properties(agent_sut PROPERTIES OUTPUT_NAME "agent_sut")

# (Opsiyonel) GUI subsystem:
# set_target_properties(agent_sut PROPERTIES WIN32_EXECUTABLE YES)


#Nasıl build edilir?
#cd agent-sut
#cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#cmake --build build --config Release
#.\build\Release\agent_sut.exe


#Çalışma beklenen endpoint’ler

#GET http://127.0.0.1:18080/healthz

#POST http://127.0.0.1:18080/action

#POST http://127.0.0.1:18080/state