cmake_minimum_required(VERSION 3.10)
project(surgical_hand_estimation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(surgical_hand_msgs REQUIRED)

# Core estimator library: no ROS dependencies, reusable inside the future
# real-hardware ros2_control plugin.
add_library(tension_estimator src/tension_estimator.cpp)
target_include_directories(tension_estimator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ROS 2 node.
add_executable(tension_estimator_node src/tension_estimator_node.cpp)
target_link_libraries(tension_estimator_node tension_estimator)
ament_target_dependencies(tension_estimator_node
  rclcpp
  std_msgs
  surgical_hand_msgs
)

install(TARGETS tension_estimator
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS tension_estimator_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_tension_estimator test/test_tension_estimator.cpp)
  target_link_libraries(test_tension_estimator tension_estimator)
endif()

ament_package()
