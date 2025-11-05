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

RUN apt-get update && apt-get install -y vim git cmake build-essential pkg-config \
    coinor-libipopt-dev freeglut3-dev libturbojpeg0-dev libglfw3-dev freenect \
    libatlas-base-dev libdrm-dev libfreenect-demos less libcamera-dev \
    libcap-dev libeigen3-dev libfreenect-dev libfmt-dev libopenjp2-7 \
    libpcl-dev libyaml-cpp-dev libyaml-cpp-dev python3-colcon-common-extensions \
    python3-pip python3-serial python3-opencv \
    ros-dev-tools ros-jazzy-rviz2 ros-jazzy-camera-info-manager ros-jazzy-cv-bridge \
    ros-jazzy-depth-image-proc ros-jazzy-image-tools ros-jazzy-pcl-ros \
    ros-jazzy-pcl-conversions ros-jazzy-geometry-msgs ros-jazzy-sensor-msgs \
    ros-jazzy-launch* ros-jazzy-irobot-create-msgs ros-jazzy-nav2-map-server \
    ffmpeg clang meson ninja-build pkg-config libyaml-dev \
    python3-yaml python3-ply python3-jinja2 openssl \
    libdw-dev libunwind-dev libudev-dev libudev-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libpython3-dev pybind11-dev libevent-dev libtiff-dev qt6-base-dev qt6-tools-dev-tools liblttng-ust-dev \
    python3-jinja2 python3-libcamera lttng-tools libexif-dev libjpeg-dev pybind11-dev libevent-dev libgtest-dev abi-compliance-checker \
    ffmpeg libavcodec-extra libavcodec-dev libavdevice-dev libpng-dev libpng-tools libepoxy-dev qt5-qmake qtmultimedia5-dev \
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
    ldconfig

RUN git clone https://github.com/raspberrypi/libcamera.git && \
    cd libcamera && \
    meson setup build --buildtype=release -Dpipelines=rpi/vc4,rpi/pisp \
    -Dipas=rpi/vc4,rpi/pisp -Dv4l2=true -Dgstreamer=enabled -Dtest=false \
    -Dlc-compliance=disabled -Dcam=disabled -Dqcam=disabled -Ddocumentation=disabled \
    -Dpycamera=enabled && \
    ninja -C build install && \
    sudo ninja -C build install

RUN git clone https://github.com/raspberrypi/rpicam-apps.git && \
    cd rpicam-apps && \
    sudo apt install cmake libboost-program-options-dev libdrm-dev libexif-dev && \
    git reset --hard d67bc17 && \
    sudo apt install qt5-qmake qtmultimedia5-dev && \
    meson setup build -Denable_libav=enabled -Denable_drm=enabled -Denable_egl=enabled -Denable_qt=enabled \
    -Denable_opencv=disabled -Denable_tflite=disabled -Denable_hailo=disabled && \
    meson compile -C build && \
    sudo meson install -C build && \
    cd / && rm -rf /tmp/libfreenect /tmp/libcamera /tmp/rpicam-apps

ENV ROS_DOMAIN_ID=1

WORKDIR /home/ws
RUN GIT_TERMINAL_PROMPT=0 git clone https://github.com/fadlio/kinect_ros2.git src/kinect_ros2
RUN sed -i 's/cv_bridge.h/cv_bridge.hpp/g' src/kinect_ros2/include/kinect_ros2/kinect_ros2_component.hpp

USER $USERNAME

RUN echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
RUN echo 'source install/setup.bash' >> ~/.bashrc
RUN sudo ldconfig

# RUN source install/setup.bash

# SHELL ["/bin/bash", "-c"]
# RUN echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc && \
#     echo 'source /ros2_ws/install/setup.bash' >> ~/.bashrc

# COPY entrypoint.sh /entrypoint.sh

# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["/bin/bash"]