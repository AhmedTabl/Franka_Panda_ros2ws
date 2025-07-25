// Copyright (c) 2021 Franka Emika GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "franka_example_controllers/comless/cartesian_velocity_example_controller.hpp"

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
CartesianVelocityExampleController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.push_back("ee_cartesian_velocity/tx");
  config.names.push_back("ee_cartesian_velocity/ty");
  config.names.push_back("ee_cartesian_velocity/tz");
  config.names.push_back("ee_cartesian_velocity/omega_x");
  config.names.push_back("ee_cartesian_velocity/omega_y");
  config.names.push_back("ee_cartesian_velocity/omega_z");

  return config;
}

controller_interface::InterfaceConfiguration
CartesianVelocityExampleController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i < 10; ++i) {
    config.names.push_back("ee_cartesian_velocity/0" + std::to_string(i));
  }
  for (int i = 10; i < 16; ++i) {
    config.names.push_back("ee_cartesian_velocity/" + std::to_string(i));
  }
  
  return config;
}

controller_interface::return_type CartesianVelocityExampleController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  // updateJointStates();
  init_time_ = init_time_ + period;
  double time_max = 4.0;
  double v_max = 0.05;
  double angle = M_PI / 4.0;
  double cycle = std::floor(
      pow(-1.0, (init_time_.seconds() - std::fmod(init_time_.seconds(), time_max)) / time_max));
  double v = cycle * v_max / 2.0 * (1.0 - std::cos(2.0 * M_PI / time_max * init_time_.seconds()));
  double v_x = std::cos(angle) * v;
  double v_z = -std::sin(angle) * v;
  std::array<double, 6> command = {{v_x, 0.0, v_z, 0.0, 0.0, 0.0}};
  for(int i = 0; i < 6; i++){
    command_interfaces_[i].set_value(command[i]);
  }
  return controller_interface::return_type::OK;
}

CallbackReturn CartesianVelocityExampleController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianVelocityExampleController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianVelocityExampleController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // updateJointStates();
  initial_q_ = q_;
  start_time_ = this->get_node()->now();
  init_time_ = rclcpp::Duration(0, 0);
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianVelocityExampleController::on_error(
  const rclcpp_lifecycle::State& /*previous_state*/){
    RCLCPP_ERROR(this->get_node()->get_logger(), "error encountered!");
    return CallbackReturn::ERROR;
  }

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianVelocityExampleController,
                       controller_interface::ControllerInterface)