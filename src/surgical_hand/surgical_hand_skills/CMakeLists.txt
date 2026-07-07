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

add_executable(hand_pose_cli src/hand_pose_cli.cpp)
ament_target_dependencies(hand_pose_cli
  rclcpp
  std_msgs
  ament_index_cpp
  yaml_cpp_vendor
)

install(TARGETS hand_pose_cli
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
