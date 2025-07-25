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

#include "franka_example_controllers/comless/joint_velocity_example_controller.hpp"

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
JointVelocityExampleController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointVelocityExampleController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::return_type JointVelocityExampleController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
//   updateJointStates();
  init_time_ = init_time_ + period;
  double omega = 0.1 * std::sin(init_time_.seconds());
  for(int i = 3; i < 7; i++){
    command_interfaces_[i].set_value(omega);
  }
  return controller_interface::return_type::OK;
}

CallbackReturn JointVelocityExampleController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointVelocityExampleController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointVelocityExampleController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  initial_q_ = q_;
  start_time_ = this->get_node()->now();
  init_time_ = rclcpp::Duration(0, 0);
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointVelocityExampleController::on_error(
  const rclcpp_lifecycle::State& /*previous_state*/){
    RCLCPP_ERROR(this->get_node()->get_logger(), "error encountered!");
    return CallbackReturn::ERROR;
  }

void JointVelocityExampleController::updateJointStates() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointVelocityExampleController,
                       controller_interface::ControllerInterface)