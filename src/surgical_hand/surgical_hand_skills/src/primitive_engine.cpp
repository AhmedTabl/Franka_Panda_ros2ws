#include "surgical_hand_skills/primitive_engine.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace surgical_hand_skills {

std::vector<Primitive> loadPrimitives(const std::string& primitives_yaml,
                                      const PoseLibrary& library) {
  const YAML::Node config = YAML::LoadFile(primitives_yaml);
  std::vector<Primitive> primitives;

  for (const auto& entry : config["primitives"]) {
    Primitive primitive;
    primitive.name = entry.first.as<std::string>();
    const YAML::Node& body = entry.second;
    primitive.description = body["description"] ? body["description"].as<std::string>() : "";

    for (const auto& step_node : body["steps"]) {
      PrimitiveStep step;
      step.name = step_node["name"].as<std::string>();
      step.pose = step_node["pose"].as<std::string>();
      if (!library.hasPose(step.pose)) {
        throw std::runtime_error("primitive '" + primitive.name + "' step '" + step.name +
                                 "' references unknown pose '" + step.pose + "'");
      }
      if (step_node["duration_s"]) {
        step.duration_s = step_node["duration_s"].as<double>();
      }
      if (step.duration_s <= 0.0) {
        throw std::runtime_error("primitive '" + primitive.name + "' step '" + step.name +
                                 "' has non-positive duration");
      }
      if (step_node["overrides"]) {
        for (const auto& value : step_node["overrides"]) {
          step.overrides[value.first.as<std::string>()] = value.second.as<double>();
        }
      }
      if (step_node["max_tension_n"]) {
        step.max_tension_n = step_node["max_tension_n"].as<double>();
      }
      if (step_node["require_contact"]) {
        step.require_contact = step_node["require_contact"].as<bool>();
      }
      // Validate override aliases (and clamping behavior) up front so a bad
      // primitives file fails at load, not mid-sequence.
      library.buildCommand(step.pose, step.overrides);
      primitive.steps.push_back(step);
    }
    if (primitive.steps.empty()) {
      throw std::runtime_error("primitive '" + primitive.name + "' has no steps");
    }
    primitives.push_back(primitive);
  }
  return primitives;
}

}  // namespace surgical_hand_skills
