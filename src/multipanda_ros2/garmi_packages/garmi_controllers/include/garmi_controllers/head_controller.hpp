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

#pragma once

#include <string>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/std_msgs/msg/float64_multi_array.hpp>


#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::JointState;
using std_msgs::msg::Float64MultiArray;


namespace garmi_controllers {

class HeadController : public controller_interface::ControllerInterface {
  public:
    controller_interface::InterfaceConfiguration command_interface_configuration() const override;
    controller_interface::InterfaceConfiguration state_interface_configuration() const override;
    controller_interface::return_type update(const rclcpp::Time& time,
                                            const rclcpp::Duration& period) override;
    CallbackReturn on_init() override;
    CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
    CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  private:
  
    // basic parameters
    std::string robot_id_ = "garmi_head";
    bool is_sim_ = true;
    const int num_joints = 2;

    // head parameters
    double k_p_ = 200.0;
    double k_p_target_ = 200.0;
    double k_d_ = 5.0;
    double k_d_target_ = 5.0;
    double pt1_filter_ = 0.001;
    double pt1_filter_target_ = 0.001;
    std::vector<double> theta_goal_{0.0,0.0};
    std::vector<double> theta_d_{0.0,0.0};
    std::vector<double> theta_curr_{0.0,0.0};
    std::vector<double> theta_prev_{0.0, 0.0};

    std::shared_ptr<rclcpp::Subscription<Float64MultiArray>> goal_pos_sub_;
    std::shared_ptr<rclcpp::Publisher<JointState>> joint_state_pub_;
    std::shared_ptr<rclcpp::Publisher<Odometry>> head_odom_pub_;

    rclcpp::Time current_time_, last_time_, last_command_time_;
    rclcpp::Clock clock_;

    
    void command_listener(const Float64MultiArray pan_tilt) {
        theta_goal_[0] = std::min(0.6, std::max(-0.6, pan_tilt.data[0]));
        theta_goal_[1] = std::min(0.2, std::max(-0.2, pan_tilt.data[1]));
    }
  };

}  // namespace franka_example_controllers