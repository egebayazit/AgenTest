cmake_minimum_required(VERSION 3.15)
project(agent-inspector)

set(CMAKE_CXX_STANDARD 17)

# Source files
add_executable(inspector
    inspector_main.cpp
    win/capture.cpp
    win/base64.cpp
)

# Include directories
target_include_directories(inspector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Windows libraries
target_link_libraries(inspector
    gdiplus
    ws2_32  # Winsock for httplib
)

## Test executable
#add_executable(test_capture
#    test_capture.cpp
#    win/capture.cpp
#    win/base64.cpp
#)

#target_include_directories(test_capture PRIVATE
#    ${CMAKE_CURRENT_SOURCE_DIR}
#)
#
#target_link_libraries(test_capture
#    gdiplus
#)
#