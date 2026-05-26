#include "franka_hardware/sim/generic_mj_joint_position_hardware_system.hpp"

#include <algorithm>
#include <cmath>
#include <exception>

#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

namespace franka_hardware {

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using CommandInterface = hardware_interface::CommandInterface;
using StateInterface = hardware_interface::StateInterface;

CallbackReturn GenericMjJointPositionHardwareSystem::on_init(
    const hardware_interface::HardwareInfo& info) {
  if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

bool GenericMjJointPositionHardwareSystem::initSim(
    rclcpp_lifecycle::LifecycleNode::SharedPtr& model_nh,
    const hardware_interface::HardwareInfo& hardware_info,
    const mjModel* m,
    mjData* d,
    unsigned int& /*update_rate*/) {
  info_ = hardware_info;
  nh_ = model_nh;
  m_ = m;
  d_ = d;

  joints_.clear();
  joints_.reserve(info_.joints.size());

  for (const auto& joint_info : info_.joints) {
    if (joint_info.command_interfaces.size() != 1 ||
        joint_info.command_interfaces[0].name != hardware_interface::HW_IF_POSITION) {
      RCLCPP_FATAL(getLogger(), "Joint '%s' must expose exactly one position command interface.",
                   joint_info.name.c_str());
      return false;
    }

    JointHandle joint;
    joint.joint_name = joint_info.name;
    joint.joint_id = mj_name2id(m_, mjOBJ_JOINT, joint.joint_name.c_str());
    if (joint.joint_id < 0) {
      RCLCPP_FATAL(getLogger(), "No MuJoCo joint named '%s' found.", joint.joint_name.c_str());
      return false;
    }
    joint.qpos_id = m_->jnt_qposadr[joint.joint_id];
    joint.qvel_id = m_->jnt_dofadr[joint.joint_id];

    auto actuator_param = joint_info.parameters.find("actuator_name");
    const std::string actuator_name =
        actuator_param == joint_info.parameters.end() ? joint.joint_name + "_actuator"
                                                      : actuator_param->second;
    joint.actuator_id = mj_name2id(m_, mjOBJ_ACTUATOR, actuator_name.c_str());
    if (joint.actuator_id < 0) {
      RCLCPP_FATAL(getLogger(), "No MuJoCo actuator named '%s' found for joint '%s'.",
                   actuator_name.c_str(), joint.joint_name.c_str());
      return false;
    }

    joint.position = d_->qpos[joint.qpos_id];
    joint.velocity = d_->qvel[joint.qvel_id];
    joint.command = joint.position;

    auto initial_position_param = joint_info.parameters.find("initial_position");
    if (initial_position_param != joint_info.parameters.end()) {
      try {
        joint.command = std::stod(initial_position_param->second);
        d_->qpos[joint.qpos_id] = joint.command;
        joint.position = joint.command;
      } catch (const std::exception& ex) {
        RCLCPP_FATAL(getLogger(), "Invalid initial_position '%s' for joint '%s': %s",
                     initial_position_param->second.c_str(), joint.joint_name.c_str(), ex.what());
        return false;
      }
    }

    d_->ctrl[joint.actuator_id] = joint.command;
    joints_.push_back(joint);
  }

  RCLCPP_INFO(getLogger(), "Initialized generic MuJoCo position hardware with %zu joints.",
              joints_.size());
  return true;
}

std::vector<StateInterface> GenericMjJointPositionHardwareSystem::export_state_interfaces() {
  std::vector<StateInterface> state_interfaces;
  state_interfaces.reserve(joints_.size() * 3);
  for (auto& joint : joints_) {
    state_interfaces.emplace_back(
        StateInterface(joint.joint_name, hardware_interface::HW_IF_POSITION, &joint.position));
    state_interfaces.emplace_back(
        StateInterface(joint.joint_name, hardware_interface::HW_IF_VELOCITY, &joint.velocity));
    state_interfaces.emplace_back(
        StateInterface(joint.joint_name, hardware_interface::HW_IF_EFFORT, &joint.effort));
  }
  return state_interfaces;
}

std::vector<CommandInterface> GenericMjJointPositionHardwareSystem::export_command_interfaces() {
  std::vector<CommandInterface> command_interfaces;
  command_interfaces.reserve(joints_.size());
  for (auto& joint : joints_) {
    command_interfaces.emplace_back(
        CommandInterface(joint.joint_name, hardware_interface::HW_IF_POSITION, &joint.command));
  }
  return command_interfaces;
}

CallbackReturn GenericMjJointPositionHardwareSystem::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  for (auto& joint : joints_) {
    joint.position = d_->qpos[joint.qpos_id];
    joint.velocity = d_->qvel[joint.qvel_id];
    joint.command = joint.position;
    d_->ctrl[joint.actuator_id] = joint.command;
  }
  RCLCPP_INFO(getLogger(), "Started generic MuJoCo position hardware.");
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type GenericMjJointPositionHardwareSystem::read(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  for (auto& joint : joints_) {
    joint.position = d_->qpos[joint.qpos_id];
    joint.velocity = d_->qvel[joint.qvel_id];
    joint.effort = d_->actuator_force[joint.actuator_id];
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type GenericMjJointPositionHardwareSystem::write(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  for (const auto& joint : joints_) {
    if (!std::isfinite(joint.command)) {
      RCLCPP_ERROR(getLogger(), "Non-finite command for joint '%s'.", joint.joint_name.c_str());
      return hardware_interface::return_type::ERROR;
    }
    d_->ctrl[joint.actuator_id] = joint.command;
  }
  return hardware_interface::return_type::OK;
}

rclcpp::Logger GenericMjJointPositionHardwareSystem::getLogger() {
  return rclcpp::get_logger("GenericMjJointPositionHardwareSystem");
}

}  // namespace franka_hardware

PLUGINLIB_EXPORT_CLASS(franka_hardware::GenericMjJointPositionHardwareSystem,
                       mujoco_ros2_control::MujocoRos2SystemInterface)
