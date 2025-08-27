// generated from rosidl_typesupport_fastrtps_c/resource/idl__rosidl_typesupport_fastrtps_c.h.em
// with input from mocap_interfaces:msg/Target.idl
// generated code does not contain a copyright notice
#ifndef MOCAP_INTERFACES__MSG__DETAIL__TARGET__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
#define MOCAP_INTERFACES__MSG__DETAIL__TARGET__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_


#include <stddef.h>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "mocap_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "mocap_interfaces/msg/detail/target__struct.h"
#include "fastcdr/Cdr.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mocap_interfaces
bool cdr_serialize_mocap_interfaces__msg__Target(
  const mocap_interfaces__msg__Target * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mocap_interfaces
bool cdr_deserialize_mocap_interfaces__msg__Target(
  eprosima::fastcdr::Cdr &,
  mocap_interfaces__msg__Target * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mocap_interfaces
size_t get_serialized_size_mocap_interfaces__msg__Target(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mocap_interfaces
size_t max_serialized_size_mocap_interfaces__msg__Target(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mocap_interfaces
bool cdr_serialize_key_mocap_interfaces__msg__Target(
  const mocap_interfaces__msg__Target * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mocap_interfaces
size_t get_serialized_size_key_mocap_interfaces__msg__Target(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mocap_interfaces
size_t max_serialized_size_key_mocap_interfaces__msg__Target(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_mocap_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, mocap_interfaces, msg, Target)();

#ifdef __cplusplus
}
#endif

#endif  // MOCAP_INTERFACES__MSG__DETAIL__TARGET__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
