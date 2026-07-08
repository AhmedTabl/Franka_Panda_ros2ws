// Surgical primitive definitions and sequencing (config side).
//
// A primitive is an ordered list of steps; each step holds a named pose,
// optional per-alias overrides (e.g. a wrist arc during needle driving),
// a duration, and OPTIONAL FEEDBACK GUARDS. Guards are parsed and carried
// now, but in this skeleton phase they are only logged/soft-checked by
// primitive_cli — closing the loop on tension/tactile feedback is the
// declared next stage of the skills slice (per the CABG task doc's FSM
// approach, arXiv:2002.02921).
//
// ROS-free for unit testing; execution/timing lives in primitive_cli.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "surgical_hand_skills/pose_library.hpp"

namespace surgical_hand_skills {

struct PrimitiveStep {
  std::string name;
  std::string pose;                       // must exist in the pose library
  Pose overrides{};                       // alias -> rad, wins over the pose
  double duration_s{1.0};
  std::optional<double> max_tension_n{};  // guard: estimated tendon tension cap
  bool require_contact{false};            // guard: fingertip contact expected
};

struct Primitive {
  std::string name;
  std::string description;
  std::vector<PrimitiveStep> steps;
};

// Load primitives from YAML and validate every referenced pose/alias
// against the library. Throws std::runtime_error on any inconsistency
// (unknown pose, unknown alias, non-positive duration, empty primitive).
std::vector<Primitive> loadPrimitives(const std::string& primitives_yaml,
                                      const PoseLibrary& library);

}  // namespace surgical_hand_skills
