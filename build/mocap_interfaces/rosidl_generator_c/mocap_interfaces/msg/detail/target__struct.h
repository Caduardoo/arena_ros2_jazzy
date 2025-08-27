// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from mocap_interfaces:msg/Target.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "mocap_interfaces/msg/target.h"


#ifndef MOCAP_INTERFACES__MSG__DETAIL__TARGET__STRUCT_H_
#define MOCAP_INTERFACES__MSG__DETAIL__TARGET__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'target_name'
#include "rosidl_runtime_c/string.h"
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.h"

/// Struct defined in msg/Target in the package mocap_interfaces.
typedef struct mocap_interfaces__msg__Target
{
  rosidl_runtime_c__String target_name;
  geometry_msgs__msg__Point position;
} mocap_interfaces__msg__Target;

// Struct for a sequence of mocap_interfaces__msg__Target.
typedef struct mocap_interfaces__msg__Target__Sequence
{
  mocap_interfaces__msg__Target * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} mocap_interfaces__msg__Target__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MOCAP_INTERFACES__MSG__DETAIL__TARGET__STRUCT_H_
