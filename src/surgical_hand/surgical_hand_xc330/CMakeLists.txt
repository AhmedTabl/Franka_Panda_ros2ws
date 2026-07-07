cmake_minimum_required(VERSION 3.10)
project(surgical_hand_xc330)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_library(xc330_client src/xc330_client.cpp)
target_include_directories(xc330_client PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(xc330_client dynamixel_sdk)

add_executable(xc330_cli src/xc330_cli.cpp)
target_link_libraries(xc330_cli xc330_client)

add_executable(xc330_state_publisher src/xc330_state_publisher.cpp)
target_link_libraries(xc330_state_publisher xc330_client)
ament_target_dependencies(xc330_state_publisher rclcpp sensor_msgs std_msgs)

install(TARGETS xc330_client
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS xc330_cli xc330_state_publisher
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(dynamixel_sdk)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_control_table test/test_control_table.cpp)
  target_include_directories(test_control_table PRIVATE include)
endif()

ament_package()
