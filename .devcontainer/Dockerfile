FROM ros:jazzy

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

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
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get install -y \ 
    build-essential \
    coinor-libipopt-dev \
    libeigen3-dev \    
    libfreenect-dev \
    libpcl-dev \
    libyaml-cpp-dev \
    libyaml-cpp-dev \
    python3-colcon-common-extensions \
    python3-pip \
    ros-jazzy-camera-info-manager \
    ros-jazzy-depth-image-proc \
    ros-jazzy-pcl-ros \
    ros-jazzy-pcl-conversions \
    ros-jazzy-geometry-msgs \
    ros-jazzy-sensor-msgs \
    ros-jazzy-launch* \
    ros-jazzy-irobot-create-msgs \
    ros-jazzy-nav2-map-server \
    coinor-libipopt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

RUN echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
RUN export ROS_DOMAIN_ID=1

WORKDIR /home/ws

# SHELL ["/bin/bash", "-c"]
# RUN echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc && \
#     echo 'source /ros2_ws/install/setup.bash' >> ~/.bashrc

# COPY entrypoint.sh /entrypoint.sh

USER $USERNAME

# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["/bin/bash"]