cmake_minimum_required(VERSION 3.10)
project(surgical_hand_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config robots
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
