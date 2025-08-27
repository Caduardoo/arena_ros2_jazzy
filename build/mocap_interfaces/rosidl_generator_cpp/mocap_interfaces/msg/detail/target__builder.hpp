// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from mocap_interfaces:msg/Target.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "mocap_interfaces/msg/target.hpp"


#ifndef MOCAP_INTERFACES__MSG__DETAIL__TARGET__BUILDER_HPP_
#define MOCAP_INTERFACES__MSG__DETAIL__TARGET__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "mocap_interfaces/msg/detail/target__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace mocap_interfaces
{

namespace msg
{

namespace builder
{

class Init_Target_position
{
public:
  explicit Init_Target_position(::mocap_interfaces::msg::Target & msg)
  : msg_(msg)
  {}
  ::mocap_interfaces::msg::Target position(::mocap_interfaces::msg::Target::_position_type arg)
  {
    msg_.position = std::move(arg);
    return std::move(msg_);
  }

private:
  ::mocap_interfaces::msg::Target msg_;
};

class Init_Target_target_name
{
public:
  Init_Target_target_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Target_position target_name(::mocap_interfaces::msg::Target::_target_name_type arg)
  {
    msg_.target_name = std::move(arg);
    return Init_Target_position(msg_);
  }

private:
  ::mocap_interfaces::msg::Target msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::mocap_interfaces::msg::Target>()
{
  return mocap_interfaces::msg::builder::Init_Target_target_name();
}

}  // namespace mocap_interfaces

#endif  // MOCAP_INTERFACES__MSG__DETAIL__TARGET__BUILDER_HPP_
