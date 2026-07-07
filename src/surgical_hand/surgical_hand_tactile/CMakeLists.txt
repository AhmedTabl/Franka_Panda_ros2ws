cmake_minimum_required(VERSION 3.10)
project(surgical_hand_tactile)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(surgical_hand_msgs REQUIRED)

add_library(tactile_source src/simulated_tactile_source.cpp)
target_include_directories(tactile_source PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(tactile_publisher src/tactile_publisher_node.cpp)
target_link_libraries(tactile_publisher tactile_source)
ament_target_dependencies(tactile_publisher rclcpp surgical_hand_msgs)

install(TARGETS tactile_source
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS tactile_publisher
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_simulated_tactile_source test/test_simulated_tactile_source.cpp)
  target_link_libraries(test_simulated_tactile_source tactile_source)
endif()

ament_package()
