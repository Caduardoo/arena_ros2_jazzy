// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from mocap_interfaces:msg/Target.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "mocap_interfaces/msg/target.hpp"


#ifndef MOCAP_INTERFACES__MSG__DETAIL__TARGET__TRAITS_HPP_
#define MOCAP_INTERFACES__MSG__DETAIL__TARGET__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "mocap_interfaces/msg/detail/target__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__traits.hpp"

namespace mocap_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Target & msg,
  std::ostream & out)
{
  out << "{";
  // member: target_name
  {
    out << "target_name: ";
    rosidl_generator_traits::value_to_yaml(msg.target_name, out);
    out << ", ";
  }

  // member: position
  {
    out << "position: ";
    to_flow_style_yaml(msg.position, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Target & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: target_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_name: ";
    rosidl_generator_traits::value_to_yaml(msg.target_name, out);
    out << "\n";
  }

  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position:\n";
    to_block_style_yaml(msg.position, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Target & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace mocap_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use mocap_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const mocap_interfaces::msg::Target & msg,
  std::ostream & out, size_t indentation = 0)
{
  mocap_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use mocap_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const mocap_interfaces::msg::Target & msg)
{
  return mocap_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<mocap_interfaces::msg::Target>()
{
  return "mocap_interfaces::msg::Target";
}

template<>
inline const char * name<mocap_interfaces::msg::Target>()
{
  return "mocap_interfaces/msg/Target";
}

template<>
struct has_fixed_size<mocap_interfaces::msg::Target>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<mocap_interfaces::msg::Target>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<mocap_interfaces::msg::Target>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MOCAP_INTERFACES__MSG__DETAIL__TARGET__TRAITS_HPP_
