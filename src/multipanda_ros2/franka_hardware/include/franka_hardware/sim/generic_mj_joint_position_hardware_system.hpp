#pragma once

#include <string>
#include <vector>

#include <hardware_interface/handle.hpp>
#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#include <mujoco/mujoco.h>
#include <mujoco_ros2_control/mujoco_ros2_control_system_interface.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp_lifecycle/state.hpp>

namespace franka_hardware {

class GenericMjJointPositionHardwareSystem
    : public mujoco_ros2_control::MujocoRos2SystemInterface {
 public:
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;

  hardware_interface::return_type read(const rclcpp::Time& time,
                                       const rclcpp::Duration& period) override;
  hardware_interface::return_type write(const rclcpp::Time& time,
                                        const rclcpp::Duration& period) override;
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_init(
      const hardware_interface::HardwareInfo& info) override;

  bool initSim(rclcpp_lifecycle::LifecycleNode::SharedPtr& model_nh,
               const hardware_interface::HardwareInfo& hardware_info,
               const mjModel* m,
               mjData* d,
               unsigned int& update_rate) override;

 private:
  struct JointHandle {
    std::string joint_name;
    int joint_id{-1};
    int qpos_id{-1};
    int qvel_id{-1};
    int actuator_id{-1};
    double command{0.0};
    double position{0.0};
    double velocity{0.0};
    double effort{0.0};
  };

  std::vector<JointHandle> joints_;
  mjData* d_{nullptr};
  const mjModel* m_{nullptr};

  static rclcpp::Logger getLogger();
};

}  // namespace franka_hardware
