#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

if [ -f $HOME/ws/install/setup.bash ]; then
    source $HOME/ws/install/setup.bash
fi

exec "$@"