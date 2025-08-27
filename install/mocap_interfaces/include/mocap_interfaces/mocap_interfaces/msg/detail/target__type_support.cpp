// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from mocap_interfaces:msg/Target.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "mocap_interfaces/msg/detail/target__functions.h"
#include "mocap_interfaces/msg/detail/target__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace mocap_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void Target_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) mocap_interfaces::msg::Target(_init);
}

void Target_fini_function(void * message_memory)
{
  auto typed_message = static_cast<mocap_interfaces::msg::Target *>(message_memory);
  typed_message->~Target();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Target_message_member_array[2] = {
  {
    "target_name",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mocap_interfaces::msg::Target, target_name),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::Point>(),  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mocap_interfaces::msg::Target, position),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Target_message_members = {
  "mocap_interfaces::msg",  // message namespace
  "Target",  // message name
  2,  // number of fields
  sizeof(mocap_interfaces::msg::Target),
  false,  // has_any_key_member_
  Target_message_member_array,  // message members
  Target_init_function,  // function to initialize message memory (memory has to be allocated)
  Target_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Target_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Target_message_members,
  get_message_typesupport_handle_function,
  &mocap_interfaces__msg__Target__get_type_hash,
  &mocap_interfaces__msg__Target__get_type_description,
  &mocap_interfaces__msg__Target__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace mocap_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<mocap_interfaces::msg::Target>()
{
  return &::mocap_interfaces::msg::rosidl_typesupport_introspection_cpp::Target_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, mocap_interfaces, msg, Target)() {
  return &::mocap_interfaces::msg::rosidl_typesupport_introspection_cpp::Target_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
