// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from mocap_interfaces:msg/Target.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "mocap_interfaces/msg/detail/target__rosidl_typesupport_introspection_c.h"
#include "mocap_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "mocap_interfaces/msg/detail/target__functions.h"
#include "mocap_interfaces/msg/detail/target__struct.h"


// Include directives for member types
// Member `target_name`
#include "rosidl_runtime_c/string_functions.h"
// Member `position`
#include "geometry_msgs/msg/point.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  mocap_interfaces__msg__Target__init(message_memory);
}

void mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_fini_function(void * message_memory)
{
  mocap_interfaces__msg__Target__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_message_member_array[2] = {
  {
    "target_name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mocap_interfaces__msg__Target, target_name),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(mocap_interfaces__msg__Target, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_message_members = {
  "mocap_interfaces__msg",  // message namespace
  "Target",  // message name
  2,  // number of fields
  sizeof(mocap_interfaces__msg__Target),
  false,  // has_any_key_member_
  mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_message_member_array,  // message members
  mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_init_function,  // function to initialize message memory (memory has to be allocated)
  mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_message_type_support_handle = {
  0,
  &mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_message_members,
  get_message_typesupport_handle_function,
  &mocap_interfaces__msg__Target__get_type_hash,
  &mocap_interfaces__msg__Target__get_type_description,
  &mocap_interfaces__msg__Target__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_mocap_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, mocap_interfaces, msg, Target)() {
  mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_message_type_support_handle.typesupport_identifier) {
    mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &mocap_interfaces__msg__Target__rosidl_typesupport_introspection_c__Target_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
