cmake_minimum_required(VERSION 3.10)
project(surgical_hand_skills)

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
find_package(ament_index_cpp REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(surgical_hand_msgs REQUIRED)

# ROS-free pose/primitive core (unit-tested).
add_library(skills_core
  src/pose_library.cpp
  src/primitive_engine.cpp
)
target_include_directories(skills_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(skills_core yaml_cpp_vendor)

add_executable(hand_pose_cli src/hand_pose_cli.cpp)
target_link_libraries(hand_pose_cli skills_core)
ament_target_dependencies(hand_pose_cli rclcpp std_msgs ament_index_cpp)

add_executable(primitive_cli src/primitive_cli.cpp)
target_link_libraries(primitive_cli skills_core)
ament_target_dependencies(primitive_cli rclcpp std_msgs ament_index_cpp surgical_hand_msgs)

install(TARGETS skills_core
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS hand_pose_cli primitive_cli
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
  ament_add_gtest(test_primitive_engine test/test_primitive_engine.cpp)
  target_link_libraries(test_primitive_engine skills_core)
endif()

ament_package()
