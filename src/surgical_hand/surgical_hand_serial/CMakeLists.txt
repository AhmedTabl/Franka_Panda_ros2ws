cmake_minimum_required(VERSION 3.10)
project(surgical_hand_serial)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)

add_library(hand_serial
  src/packet.cpp
  src/serial_port.cpp
  src/due_client.cpp
)
target_include_directories(hand_serial PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(hand_serial_cli src/hand_serial_cli.cpp)
target_link_libraries(hand_serial_cli hand_serial)

install(TARGETS hand_serial
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS hand_serial_cli
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_packet test/test_packet.cpp)
  target_link_libraries(test_packet hand_serial)
  ament_add_gtest(test_client_pty test/test_client_pty.cpp)
  target_link_libraries(test_client_pty hand_serial util)  # util: openpty
endif()

ament_package()
