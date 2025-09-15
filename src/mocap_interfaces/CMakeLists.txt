cmake_minimum_required(VERSION 3.8)
project(mocap_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# DEPENDENCIES
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Target.msg"
  "msg/Blob.msg"
  "msg/BlobArray.msg"
  DEPENDENCIES geometry_msgs
)

add_executable(webcam_node src/webcam_publisher_node.cpp)

#ament_target_dependencies(${PROJECT_NAME} cv_bridge)

ament_target_dependencies(webcam_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
)

install(TARGETS webcam_node
  DESTINATION lib/${PROJECT_NAME}
)

# Tests
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
