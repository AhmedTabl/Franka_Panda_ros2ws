#include "garmi_controllers/head_controller.hpp"
// #include "../include/garmi_controllers/head_controller.hpp"
#include <rclcpp/duration.hpp>

using std::placeholders::_1;

namespace garmi_controllers{


controller_interface::InterfaceConfiguration 
  HeadController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.push_back(robot_id_ + "_joint1/effort");
  config.names.push_back(robot_id_ + "_joint2/effort");
  return config;
};

controller_interface::InterfaceConfiguration 
  HeadController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names.push_back(robot_id_ + "_joint1/position");
  config.names.push_back(robot_id_ + "_joint1/velocity");
  config.names.push_back(robot_id_ + "_joint1/effort");
  config.names.push_back(robot_id_ + "_joint2/position");
  config.names.push_back(robot_id_ + "_joint2/velocity");
  config.names.push_back(robot_id_ + "_joint2/effort");
  return config;
};

controller_interface::return_type HeadController::update(const rclcpp::Time& time,
                                                               const rclcpp::Duration& period){
  // Get current theta values                                                                
  theta_curr_[0] = state_interfaces_.at(0).get_value(); // left-right
  theta_curr_[1] = state_interfaces_.at(3).get_value(); // down-up
  
  // PD control 
  for (size_t joint_number = 0; joint_number < 2; joint_number++) {
    
    // Get current and previous theta values
    double theta = theta_curr_[joint_number];
    double theta_previous = theta_prev_[joint_number];
    
    // Obtain velocities
    double dtheta_d = 0;
    double dtheta = (theta - theta_previous) / period.seconds();
    theta_d_[joint_number] = theta_d_[joint_number] * (1.0 - pt1_filter_) + pt1_filter_ * theta_goal_[joint_number];
    
    // Calculate the approapriate torque value - PD control    
    double tau_j = std::min(15.0, std::max(-15.0, (theta_d_[joint_number] - theta) * k_p_ + 
                                          (dtheta_d - dtheta) * k_d_));
    
    // Set command and previous theta
    command_interfaces_[joint_number].set_value(tau_j);
    theta_prev_[joint_number] = theta;

    // Set gains & filter values
    // Currently no effect - potentially target gains could be dynamically set at runtime
    k_p_ = k_p_ * (1.0 - 0.001) + k_p_target_ * 0.001;
    k_d_ = k_d_ * (1.0 - 0.001) + k_d_target_ * 0.001;
    pt1_filter_ = pt1_filter_ * (1.0 - 0.001) + pt1_filter_target_ * 0.001;
  }
  return controller_interface::return_type::OK;
};

CallbackReturn HeadController::on_init(){
  // nothing for now
  try {
    auto_declare<std::string>("robot_id", "garmi_head");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
} ;

CallbackReturn HeadController::on_configure(const rclcpp_lifecycle::State& previous_state){
  // nothing for now
  robot_id_ = get_node()->get_parameter("robot_id").as_string();
  is_sim_ = get_node()->get_parameter("sim").as_bool();
  goal_pos_sub_ = this->get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(robot_id_ + "/head_command", 1, std::bind(&HeadController::command_listener, this, _1));
  return CallbackReturn::SUCCESS;
} ;

CallbackReturn HeadController::on_activate(const rclcpp_lifecycle::State& previous_state){
  current_time_ = this->get_node()->now();
  last_time_ = this->get_node()->now();
  RCLCPP_INFO(get_node()->get_logger(), "on_activate");
  return CallbackReturn::SUCCESS;
} ;

CallbackReturn HeadController::on_deactivate(const rclcpp_lifecycle::State& previous_state){
  return CallbackReturn::SUCCESS;
} ;
}   // namespace
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(garmi_controllers::HeadController,
                       controller_interface::ControllerInterface)