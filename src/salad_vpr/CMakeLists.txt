cmake_minimum_required(VERSION 3.0.2)
project(salad_vpr)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  message_generation
  dynamic_reconfigure
)

add_message_files(
  FILES
    Embedding.msg
)

add_service_files(
  FILES
    QueryTopK.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
)

generate_dynamic_reconfigure_options(
  cfg/SaladVpr.cfg
)

catkin_package(
  CATKIN_DEPENDS
    std_msgs
    sensor_msgs
    image_transport
    cv_bridge
    message_runtime
    dynamic_reconfigure
)

catkin_install_python(
  PROGRAMS
    scripts/vpr_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.yaml"
)
