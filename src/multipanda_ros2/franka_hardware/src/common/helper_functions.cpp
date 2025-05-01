#include <franka_hardware/common/helper_functions.hpp>
// A dump of functions that are used by both hardware and multi_hardware interfaces.
namespace franka_hardware {

  bool startsWith(const std::string& mainStr, const std::string& toMatch) {
    if (mainStr.size() < toMatch.size()) {
        return false;
    }
    return mainStr.compare(0, toMatch.size(), toMatch) == 0;
  };
  
bool all_of_element_has_string(std::vector<std::string> vec, std::string content){
  if(vec.size() == 0){
    return false;
  }
  return std::all_of(
        vec.begin(), vec.end(),
        [&](std::string elem) { return elem.find(content) != std::string::npos; });
  
}
int check_command_mode_type(std::vector<std::string> interfaces){
  if(interfaces.size() != 0){
    bool is_cartesian = all_of_element_has_string(interfaces, "ee_cartesian");
    bool is_joint = all_of_element_has_string(interfaces, "joint");
    if(!(is_cartesian || is_joint)){
      return -1;
    }
    if(is_joint){
      return 1;
    }
    if(is_cartesian){
      return 2;
    }
  }
  else{
    return 0;
  }
  return -1;
}
std::string get_ns(std::string const& s)
{
    std::string::size_type pos = s.find_last_of('_');
    if (pos != std::string::npos)
    {
        return s.substr(0, pos);
    }
    else
    {
        return s;
    }
}

// Function for extracting joint number
int get_joint_no(std::string const& s){
  int no = s.back() - '0' - 1;
  return no;
}

}