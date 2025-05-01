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

#include <franka_example_controllers/comless/move_to_start_example_controller.hpp>

#include <cassert>
#include <cmath>
#include <exception>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>

namespace franka_example_controllers {

// This function defines the command interfaces that the controller will use
// to send control commands (effort/torque) to each joint of the robotic arm.
controller_interface::InterfaceConfiguration
MoveToStartExampleController::command_interface_configuration() const {
  // Create an InterfaceConfiguration object to store the interface details
  controller_interface::InterfaceConfiguration config;

  // Set the configuration type to INDIVIDUAL, meaning each joint is configured separately.
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Loop through all joints (assumed to be numbered from 1 to num_joints)
  for (int i = 1; i <= num_joints; ++i) {
    // Construct the name of the effort (torque) interface for each joint
    // Example: "panda_joint1/effort", "panda_joint2/effort", ..., "panda_joint7/effort"
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }

  // Return the configured command interface
  return config;
}

// This function defines the state interfaces that the controller will read from
// to get the current state (position & velocity) of each joint in the robotic arm.
controller_interface::InterfaceConfiguration
MoveToStartExampleController::state_interface_configuration() const {
  // Create an InterfaceConfiguration object to store the state interface details
  controller_interface::InterfaceConfiguration config;

  // Set the configuration type to INDIVIDUAL, meaning each joint has its own separate state interfaces.
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  // Loop through all joints (assumed to be numbered from 1 to num_joints)
  for (int i = 1; i <= num_joints; ++i) {
    // Add the position state interface for each joint
    // Example: "panda_joint1/position", "panda_joint2/position", ..., "panda_joint7/position"
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");

    // Add the velocity state interface for each joint
    // Example: "panda_joint1/velocity", "panda_joint2/velocity", ..., "panda_joint7/velocity"
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }

  // Return the configured state interface
  return config;
}


// This function is called periodically by the controller to update the robot's state
// and compute the necessary control commands for moving the arm.
controller_interface::return_type MoveToStartExampleController::update(
    const rclcpp::Time& /*time*/,  // The current time (unused)
    const rclcpp::Duration& /*period*/) {  // The time step since the last update (unused)

  // Get the current joint states (positions & velocities) from the state interfaces
  updateJointStates();

  // Calculate the elapsed time since the start of the trajectory
  auto trajectory_time = this->get_node()->now() - start_time_;

  // Get the desired joint positions at the current trajectory time
  auto motion_generator_output = motion_generator_->getDesiredJointPositions(trajectory_time);
  
  // Extract the target joint positions and check if the trajectory is finished
  Vector7d q_desired = motion_generator_output.first;
  bool finished = motion_generator_output.second;

  // If the motion is not yet finished, compute the control commands
  if (not finished) {
    const double kAlpha = 0.99;  // Low-pass filter constant for velocity smoothing

    // Apply a low-pass filter to smooth out the velocity signal
    dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;

    // Compute the desired torques using proportional-derivative (PD) control
    Vector7d tau_d_calculated =
        k_gains_.cwiseProduct(q_desired - q_) +  // Proportional term (position error)
        d_gains_.cwiseProduct(-dq_filtered_);   // Derivative term (velocity damping)

    // Send the computed torque commands to the robot's actuators
    for (int i = 0; i < 7; ++i) {
      command_interfaces_[i].set_value(tau_d_calculated(i));
    }
  } else {
    // If the motion is finished, set all torque commands to zero to stop movement
    for (auto& command_interface : command_interfaces_) {
      command_interface.set_value(0);
    }
  }

  // Return success to indicate the update cycle completed correctly
  return controller_interface::return_type::OK;
}


// This function is called when the controller is initialized.
// It sets up the default goal joint positions and declares necessary parameters.
CallbackReturn MoveToStartExampleController::on_init() {
  // Define the default joint goal positions (in radians)
  // This represents a predefined "ready" or "home" position for the robot arm.
  q_goal_ << 0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4; //start position
  //q_goal_ << -0.3082215861029207, 1.0457879639801027, 0.28633301698937574, -1.301182024119404, 1.1598439586038645, 2.5076533336175815, -0.5344755916886573; //test position

  try {
    // Declare ROS 2 parameters with default values.
    // These parameters can be set externally (e.g., via a launch file or dynamic reconfigure).
    
    // The "arm_id" parameter identifies the robot arm (default: "panda").
    auto_declare<std::string>("arm_id", "panda");

    // The "k_gains" parameter is a vector for proportional control gains (default: empty).
    auto_declare<std::vector<double>>("k_gains", {});

    // The "d_gains" parameter is a vector for derivative control gains (default: empty).
    auto_declare<std::vector<double>>("d_gains", {});
  } catch (const std::exception& e) {
    // If an error occurs during parameter declaration, print an error message and return failure.
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }

  // Return success if initialization completes without errors.
  return CallbackReturn::SUCCESS;
}


// This function is called when the controller enters the "configured" state.
// It retrieves parameters and checks if they are valid.
CallbackReturn MoveToStartExampleController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  
  // Get the "arm_id" parameter from the ROS 2 node.
  arm_id_ = get_node()->get_parameter("arm_id").as_string();

  // Retrieve the "k_gains" (stiffness gains) and "d_gains" (damping gains) parameters.
  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();

  // Check if the k_gains parameter is set; if not, log a fatal error and return failure.
  if (k_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains parameter not set");
    return CallbackReturn::FAILURE;
  }

  // Ensure k_gains has the correct number of values (one per joint).
  if (k_gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains should be of size %d but is of size %ld",
                 num_joints, k_gains.size());
    return CallbackReturn::FAILURE;
  }

  // Check if the d_gains parameter is set; if not, log a fatal error and return failure.
  if (d_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains parameter not set");
    return CallbackReturn::FAILURE;
  }

  // Ensure d_gains has the correct number of values (one per joint).
  if (d_gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains should be of size %d but is of size %ld",
                 num_joints, d_gains.size());
    return CallbackReturn::FAILURE;
  }

  // Assign the retrieved gains to the class variables.
  for (int i = 0; i < num_joints; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
  }

  // Reset the filtered velocity variable to zero.
  dq_filtered_.setZero();

  // If everything is valid, return success.
  return CallbackReturn::SUCCESS;
}


// This function is called when the controller transitions from "inactive" to "active".
// It initializes motion generation and starts tracking time.
CallbackReturn MoveToStartExampleController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  
  // Update the current joint states (position and velocity).
  updateJointStates();

  // Create a new motion generator instance.
  // The MotionGenerator takes:
  // - A velocity scaling factor (0.2)
  // - The current joint positions (q_)
  // - The target joint positions (q_goal_)
  motion_generator_ = std::make_unique<MotionGenerator>(0.2, q_, q_goal_);

  // Store the current time as the start time for motion execution.
  start_time_ = this->get_node()->now();

  // Indicate successful activation.
  return CallbackReturn::SUCCESS;
}


// This function updates the current joint states (position and velocity).
void MoveToStartExampleController::updateJointStates() {
  // Loop through each joint (num_joints is the total number of joints).
  for (auto i = 0; i < num_joints; ++i) {
    
    // Access the position and velocity interfaces for the i-th joint.
    // state_interfaces_ is a container that holds the joint interfaces.
    const auto& position_interface = state_interfaces_.at(2 * i);    // Position interface for joint i.
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1); // Velocity interface for joint i.

    // Assert that the interface names are correct (they should be "position" and "velocity").
    // This ensures that the correct interfaces are being accessed.
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    // Update the joint position and velocity based on the interface values.
    // q_ stores the joint positions (q_ is a Vector7d).
    q_(i) = position_interface.get_value();
    
    // dq_ stores the joint velocities (dq_ is a Vector7d).
    dq_(i) = velocity_interface.get_value();
  }
}

}  // namespace franka_example_controllers

//Adds this controller as a ros2 plugin
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::MoveToStartExampleController,
                       controller_interface::ControllerInterface)
