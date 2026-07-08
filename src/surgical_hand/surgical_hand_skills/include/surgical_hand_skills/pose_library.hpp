// Named-pose handling shared by hand_pose_cli and the primitive engine.
//
// Loads the joint set (surgical_hand_description/config/hand_joints.yaml,
// the single source of truth) and the named poses (config/named_poses.yaml,
// keyed by joint ALIAS), and builds command vectors in joint-config order
// with limit clamping. ROS-free for unit testing.

#pragma once

#include <map>
#include <string>
#include <vector>

namespace surgical_hand_skills {

struct HandJoint {
  std::string name;   // URDF joint name (command vector order source)
  std::string alias;  // stable semantic name used by poses/primitives
  double lower{0.0};
  double upper{0.0};
  double initial_position{0.0};
};

using Pose = std::map<std::string, double>;  // alias -> position [rad]

class PoseLibrary {
 public:
  // Throws std::runtime_error on unreadable/malformed files.
  PoseLibrary(const std::string& hand_joints_yaml, const std::string& named_poses_yaml);

  bool hasPose(const std::string& name) const { return poses_.count(name) > 0; }
  std::vector<std::string> poseNames() const;
  const std::vector<HandJoint>& joints() const { return joints_; }

  // Command vector in joint order: pose values by alias, `overrides` (also
  // by alias) taking precedence, unspecified joints at initial_position.
  // Values are clamped to joint limits; clamping is reported through
  // `clamped_aliases` when provided. Throws std::out_of_range on an unknown
  // pose and std::invalid_argument on an unknown alias in overrides.
  std::vector<double> buildCommand(const std::string& pose_name,
                                   const Pose& overrides = {},
                                   std::vector<std::string>* clamped_aliases = nullptr) const;

 private:
  std::vector<HandJoint> joints_;
  std::map<std::string, Pose> poses_;
};

}  // namespace surgical_hand_skills
