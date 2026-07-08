// Unit tests for PoseLibrary and the primitive loader, using small YAML
// fixtures written to a temp dir (no dependence on installed shares).

#include <cstdio>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "surgical_hand_skills/pose_library.hpp"
#include "surgical_hand_skills/primitive_engine.hpp"

namespace shs = surgical_hand_skills;

namespace {

class Fixture : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = ::testing::TempDir();
    write("joints.yaml", R"(
joints:
  - {name: J1, alias: wrist, lower: -1.0, upper: 0.5, initial_position: 0.0}
  - {name: J2, alias: index_mcp, lower: -0.4, upper: 1.7, initial_position: 0.1}
  - {name: J3, alias: thumb_mcp, lower: -0.4, upper: 1.7, initial_position: 0.0}
)");
    write("poses.yaml", R"(
poses:
  open: {}
  grasp: {index_mcp: 0.9, thumb_mcp: 0.8}
  too_big: {index_mcp: 5.0}
)");
  }

  void write(const std::string& name, const std::string& content) {
    std::ofstream(path(name)) << content;
  }
  std::string path(const std::string& name) const { return dir_ + "/" + name; }

  shs::PoseLibrary library() const { return {path("joints.yaml"), path("poses.yaml")}; }

  std::string dir_;
};

}  // namespace

TEST_F(Fixture, BuildCommandOrderAndDefaults) {
  const auto command = library().buildCommand("grasp");
  ASSERT_EQ(command.size(), 3u);
  EXPECT_DOUBLE_EQ(command[0], 0.0);  // wrist: initial_position
  EXPECT_DOUBLE_EQ(command[1], 0.9);  // index_mcp from pose
  EXPECT_DOUBLE_EQ(command[2], 0.8);  // thumb_mcp from pose
}

TEST_F(Fixture, OverridesWinOverPoseAndDefaults) {
  const auto command = library().buildCommand("grasp", {{"index_mcp", 0.2}, {"wrist", -0.5}});
  EXPECT_DOUBLE_EQ(command[0], -0.5);
  EXPECT_DOUBLE_EQ(command[1], 0.2);
  EXPECT_DOUBLE_EQ(command[2], 0.8);  // untouched
}

TEST_F(Fixture, ClampingReportsAliases) {
  std::vector<std::string> clamped;
  const auto command = library().buildCommand("too_big", {}, &clamped);
  EXPECT_DOUBLE_EQ(command[1], 1.7);  // clamped to upper
  ASSERT_EQ(clamped.size(), 1u);
  EXPECT_EQ(clamped[0], "index_mcp");
}

TEST_F(Fixture, UnknownPoseAndAliasThrow) {
  EXPECT_THROW(library().buildCommand("nope"), std::out_of_range);
  EXPECT_THROW(library().buildCommand("open", {{"bogus_alias", 1.0}}), std::invalid_argument);
}

TEST_F(Fixture, PoseWithUnknownAliasRejectedAtLoad) {
  write("bad_poses.yaml", "poses:\n  broken: {no_such_joint: 1.0}\n");
  EXPECT_THROW(shs::PoseLibrary(path("joints.yaml"), path("bad_poses.yaml")),
               std::runtime_error);
}

TEST_F(Fixture, PrimitivesParseWithGuardsAndOverrides) {
  write("primitives.yaml", R"(
primitives:
  demo:
    description: two-step demo
    steps:
      - {name: a, pose: open, duration_s: 0.5}
      - name: b
        pose: grasp
        duration_s: 2.0
        overrides: {wrist: -0.3}
        max_tension_n: 1.5
        require_contact: true
)");
  const auto primitives = shs::loadPrimitives(path("primitives.yaml"), library());
  ASSERT_EQ(primitives.size(), 1u);
  const auto& demo = primitives[0];
  EXPECT_EQ(demo.description, "two-step demo");
  ASSERT_EQ(demo.steps.size(), 2u);
  EXPECT_EQ(demo.steps[0].pose, "open");
  EXPECT_FALSE(demo.steps[0].max_tension_n.has_value());
  EXPECT_DOUBLE_EQ(demo.steps[1].overrides.at("wrist"), -0.3);
  EXPECT_DOUBLE_EQ(*demo.steps[1].max_tension_n, 1.5);
  EXPECT_TRUE(demo.steps[1].require_contact);
}

TEST_F(Fixture, PrimitiveValidationFailures) {
  write("bad1.yaml",
        "primitives:\n  p:\n    steps:\n      - {name: a, pose: nope, duration_s: 1.0}\n");
  EXPECT_THROW(shs::loadPrimitives(path("bad1.yaml"), library()), std::runtime_error);

  write("bad2.yaml",
        "primitives:\n  p:\n    steps:\n      - {name: a, pose: open, duration_s: 0.0}\n");
  EXPECT_THROW(shs::loadPrimitives(path("bad2.yaml"), library()), std::runtime_error);

  write("bad3.yaml",
        "primitives:\n  p:\n    steps:\n      - name: a\n        pose: open\n"
        "        duration_s: 1.0\n        overrides: {bogus: 1.0}\n");
  EXPECT_THROW(shs::loadPrimitives(path("bad3.yaml"), library()), std::invalid_argument);

  write("bad4.yaml", "primitives:\n  p:\n    steps: []\n");
  EXPECT_THROW(shs::loadPrimitives(path("bad4.yaml"), library()), std::runtime_error);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
