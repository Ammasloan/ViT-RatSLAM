#!/usr/bin/env bash
set -e

source /opt/ros/melodic/setup.bash
if [ -f /opt/catkin_ws/devel/setup.bash ]; then
  source /opt/catkin_ws/devel/setup.bash
fi

exec "$@"
