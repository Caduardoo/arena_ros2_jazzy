FROM ros:humble-ros-base as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git pkg-config cmake g++ python3-pip \
    libgnutls28-dev openssl libtiff-dev libevent-dev \
    libdrm-dev python3-yaml python3-ply python3-jinja2 \
    libboost-program-options-dev libexif-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libpng-dev libavcodec-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install meson ninja

WORKDIR /usr/src
RUN git clone https://github.com/raspberrypi/libcamera.git && \
    cd libcamera && \
    git checkout libcamera-v0.3.0-rpi && \
    meson setup build -Dpycamera=enabled -Ddocumentation=disabled && \
    ninja -C build && \
    ninja -C build install

WORKDIR /usr/src
RUN git clone https://github.com/raspberrypi/rpicam-apps.git && \
    cd rpicam-apps && \
    git checkout v1.9.0 && \
    meson setup build && \
    ninja -C build && \
    ninja -C build install

FROM ros:humble-ros-base

ARG USERNAME=ieeerasufcg
ENV USERNAME=${USERNAME}
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ARG HOST_VIDEO_GID

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

COPY --from=builder /usr/local/ /usr/local/
RUN ldconfig

RUN apt-get update && apt-get install -y --no-install-recommends \ 
    vim git cmake build-essential pkg-config less ros-dev-tools \
    # Math/Vision libs
    coinor-libipopt-dev freeglut3-dev libturbojpeg0-dev libglfw3-dev \ 
    libatlas-base-dev libeigen3-dev libpcl-dev \
    # Hardware libs
    freenect libfreenect-dev libfreenect-demos \
    libcamera-dev python3-libcamera libcamera-apps \
    # Misc libs
    libdrm-dev libcap-dev libfmt-dev libopenjp2-7 libyaml-cpp-dev \
    # Python tools
    python3-colcon-common-extensions python3-pip python3-serial \
    # ROS packages
    ros-dev-tools ros-humble-rviz2 ros-humble-camera-info-manager ros-humble-depth-image-proc \
    ros-humble-image-tools ros-humble-pcl-ros ros-humble-pcl-conversions \
    ros-humble-geometry-msgs ros-humble-sensor-msgs ros-humble-launch* \
    ros-humble-irobot-create-msgs ros-humble-nav2-map-server \
    ## libkms++-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /tmp/build_libfreenect
RUN GIT_TERMINAL_PROMPT=0 git clone https://github.com/OpenKinect/libfreenect.git . && \
    mkdir build && cd build && \
    # flag to not build libfreenect -DBUILD_EXAMPLES=OFF -DBUILD_REDIST_PACKAGE=OFF
    cmake .. && \
    make && \
    make install && \
    ldconfig && \
    rm -rf /tmp/build_libfreenect

ENV ROS_DOMAIN_ID=1
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

WORKDIR /home/${USERNAME}/ws
RUN GIT_TERMINAL_PROMPT=0 git clone https://github.com/fadlio/kinect_ros2.git src/kinect_ros2
RUN sed -i 's/cv_bridge.h/cv_bridge.hpp/g' src/kinect_ros2/include/kinect_ros2/kinect_ros2_component.hpp

WORKDIR /home/VirtualMoCap

COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

USER $USERNAME

COPY entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]