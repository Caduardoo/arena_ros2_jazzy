FROM ros:jazzy

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ARG HOST_VIDEO_GID

# RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

RUN apt-get update && apt-get install -y sudo \
    && if id -u $USER_UID > /dev/null 2>&1; then \ 
        EXISTING_USER=$(id -un $USER_UID); \
        if [ "${EXISTING_USER}" != "${USERNAME}" ]; then \
            userdel -r -f $EXISTING_USER; \
        fi; \
    fi \
    && if ! getent group $USER_GID > /dev/null; then \
        groupadd -g $USER_GID $USERNAME; \
    fi \  
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME --shell /bin/bash \
    && if [ ! -z "$HOST_VIDEO_GID" ]; then \
        if getent group video; then groupmod -g $HOST_VIDEO_GID video; else groupadd --gid $HOST_VIDEO_GID video; fi && \
        usermod -a -G video $USERNAME; \
       fi \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get install -y \ 
    vim git cmake build-essential pkg-config \
    coinor-libipopt-dev \
    freeglut3-dev \
    libturbojpeg0-dev \
    libglfw3-dev \ 
    freenect \
    libfreenect-demos \
    less \
    libcap-dev \
    libeigen3-dev \    
    libfreenect-dev \
    libpcl-dev \
    libyaml-cpp-dev \
    libyaml-cpp-dev \
    python3-colcon-common-extensions \
    python3-pip \
    python3-serial \
    python3-opencv \
    ros-dev-tools \
    ros-jazzy-rviz2 \
    ros-jazzy-camera-info-manager \
    ros-jazzy-depth-image-proc \
    ros-jazzy-image-tools \
    ros-jazzy-pcl-ros \
    ros-jazzy-pcl-conversions \
    ros-jazzy-geometry-msgs \
    ros-jazzy-sensor-msgs \
    ros-jazzy-launch* \
    ros-jazzy-irobot-create-msgs \
    ros-jazzy-nav2-map-server \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

WORKDIR /tmp
RUN GIT_TERMINAL_PROMPT=0 git clone https://github.com/OpenKinect/libfreenect.git && \
    cd libfreenect && \ 
    mkdir build && cd build && \
    # flag to not build libfreenect -DBUILD_EXAMPLES=OFF -DBUILD_REDIST_PACKAGE=OFF
    cmake .. && \
    make && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/libfreenect

COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

ENV ROS_DOMAIN_ID=1

WORKDIR /home/ws
RUN GIT_TERMINAL_PROMPT=0 git clone https://github.com/fadlio/kinect_ros2.git src/kinect_ros2
RUN sed -i 's/cv_bridge.h/cv_bridge.hpp/g' src/kinect_ros2/include/kinect_ros2/kinect_ros2_component.hpp

WORKDIR /home/VirtualMoCap
COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

USER $USERNAME

RUN echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc

# SHELL ["/bin/bash", "-c"]
# RUN echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc && \
#     echo 'source /ros2_ws/install/setup.bash' >> ~/.bashrc

# COPY entrypoint.sh /entrypoint.sh

# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["/bin/bash"]