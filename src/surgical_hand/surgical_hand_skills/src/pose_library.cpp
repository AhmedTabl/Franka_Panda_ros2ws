#include "surgical_hand_skills/pose_library.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace surgical_hand_skills {

PoseLibrary::PoseLibrary(const std::string& hand_joints_yaml,
                         const std::string& named_poses_yaml) {
  const YAML::Node joints_config = YAML::LoadFile(hand_joints_yaml);
  for (const auto& entry : joints_config["joints"]) {
    HandJoint joint;
    joint.name = entry["name"].as<std::string>();
    joint.alias = entry["alias"].as<std::string>();
    joint.lower = entry["lower"].as<double>();
    joint.upper = entry["upper"].as<double>();
    joint.initial_position = entry["initial_position"].as<double>();
    joints_.push_back(joint);
  }
  if (joints_.empty()) {
    throw std::runtime_error("no joints found in " + hand_joints_yaml);
  }

  std::set<std::string> known_aliases;
  for (const auto& joint : joints_) {
    known_aliases.insert(joint.alias);
  }

  const YAML::Node poses_config = YAML::LoadFile(named_poses_yaml);
  for (const auto& pose_entry : poses_config["poses"]) {
    const auto pose_name = pose_entry.first.as<std::string>();
    Pose pose;
    for (const auto& value : pose_entry.second) {
      const auto alias = value.first.as<std::string>();
      if (known_aliases.count(alias) == 0) {
        throw std::runtime_error("pose '" + pose_name + "' uses unknown joint alias '" +
                                 alias + "'");
      }
      pose[alias] = value.second.as<double>();
    }
    poses_[pose_name] = pose;
  }
}

std::vector<std::string> PoseLibrary::poseNames() const {
  std::vector<std::string> names;
  names.reserve(poses_.size());
  for (const auto& [name, _] : poses_) {
    names.push_back(name);
  }
  return names;
}

std::vector<double> PoseLibrary::buildCommand(const std::string& pose_name,
                                              const Pose& overrides,
                                              std::vector<std::string>* clamped_aliases) const {
  const auto pose_it = poses_.find(pose_name);
  if (pose_it == poses_.end()) {
    throw std::out_of_range("unknown pose: " + pose_name);
  }
  for (const auto& [alias, _] : overrides) {
    if (std::none_of(joints_.begin(), joints_.end(),
                     [&](const HandJoint& j) { return j.alias == alias; })) {
      throw std::invalid_argument("override uses unknown joint alias: " + alias);
    }
  }

  std::vector<double> command;
  command.reserve(joints_.size());
  for (const auto& joint : joints_) {
    double value = joint.initial_position;
    if (auto it = pose_it->second.find(joint.alias); it != pose_it->second.end()) {
      value = it->second;
    }
    if (auto it = overrides.find(joint.alias); it != overrides.end()) {
      value = it->second;
    }
    const double clamped = std::clamp(value, joint.lower, joint.upper);
    if (clamped != value && clamped_aliases != nullptr) {
      clamped_aliases->push_back(joint.alias);
    }
    command.push_back(clamped);
  }
  return command;
}

}  // namespace surgical_hand_skills
