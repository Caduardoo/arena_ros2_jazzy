// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from mocap_interfaces:msg/Target.idl
// generated code does not contain a copyright notice

#include "mocap_interfaces/msg/detail/target__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_mocap_interfaces
const rosidl_type_hash_t *
mocap_interfaces__msg__Target__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xe4, 0x20, 0x92, 0x0a, 0xc6, 0xb5, 0x65, 0xc3,
      0x4d, 0x9f, 0x41, 0xf1, 0xaa, 0x47, 0x85, 0x2a,
      0x55, 0x83, 0xc3, 0xf5, 0x84, 0x96, 0x1b, 0xf3,
      0xf8, 0x74, 0x44, 0x20, 0x7f, 0x28, 0x6f, 0x1c,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "geometry_msgs/msg/detail/point__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t geometry_msgs__msg__Point__EXPECTED_HASH = {1, {
    0x69, 0x63, 0x08, 0x48, 0x42, 0xa9, 0xb0, 0x44,
    0x94, 0xd6, 0xb2, 0x94, 0x1d, 0x11, 0x44, 0x47,
    0x08, 0xd8, 0x92, 0xda, 0x2f, 0x4b, 0x09, 0x84,
    0x3b, 0x9c, 0x43, 0xf4, 0x2a, 0x7f, 0x68, 0x81,
  }};
#endif

static char mocap_interfaces__msg__Target__TYPE_NAME[] = "mocap_interfaces/msg/Target";
static char geometry_msgs__msg__Point__TYPE_NAME[] = "geometry_msgs/msg/Point";

// Define type names, field names, and default values
static char mocap_interfaces__msg__Target__FIELD_NAME__target_name[] = "target_name";
static char mocap_interfaces__msg__Target__FIELD_NAME__position[] = "position";

static rosidl_runtime_c__type_description__Field mocap_interfaces__msg__Target__FIELDS[] = {
  {
    {mocap_interfaces__msg__Target__FIELD_NAME__target_name, 11, 11},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {mocap_interfaces__msg__Target__FIELD_NAME__position, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {geometry_msgs__msg__Point__TYPE_NAME, 23, 23},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription mocap_interfaces__msg__Target__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {geometry_msgs__msg__Point__TYPE_NAME, 23, 23},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
mocap_interfaces__msg__Target__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {mocap_interfaces__msg__Target__TYPE_NAME, 27, 27},
      {mocap_interfaces__msg__Target__FIELDS, 2, 2},
    },
    {mocap_interfaces__msg__Target__REFERENCED_TYPE_DESCRIPTIONS, 1, 1},
  };
  if (!constructed) {
    assert(0 == memcmp(&geometry_msgs__msg__Point__EXPECTED_HASH, geometry_msgs__msg__Point__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = geometry_msgs__msg__Point__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "string target_name\n"
  "\n"
  "geometry_msgs/Point position";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
mocap_interfaces__msg__Target__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {mocap_interfaces__msg__Target__TYPE_NAME, 27, 27},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 49, 49},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
mocap_interfaces__msg__Target__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[2];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 2, 2};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *mocap_interfaces__msg__Target__get_individual_type_description_source(NULL),
    sources[1] = *geometry_msgs__msg__Point__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
