# Importing modules...
import os
import time
import socket
import numpy as np
import threading
import streamlit as st
from multiprocessing import Process

from virtualmocap.plot.viewer3d import Viewer3D
from virtualmocap.vision.camera import Camera
from virtualmocap.integration.client import Client
from virtualmocap.integration.mocaprasp.server import MoCapRasp_Server

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String as RosString
    from mocap_interfaces.msg import BlobArray
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

class MocapServerNode(Node):
    def __init__(self, server_logic: MoCapRasp_Server):
        super().__init__('mocap_server_node')
        self.server = server_logic
        self.subscribers = []
        self.get_logger().info("ROS2 Mocap Server Node initialized")
        self.control_publisher = self.create_publisher(RosString, '/mocap/capture_control', 10)

    def discover_and_subscribe_clients(self):
        self.get_logger().info(f"Discovering clients... Found {len(self.server.clients)} active clients.")
        for sub in self.subscribers:
            self.destroy_subscription(sub)
        self.subscribers.clear()

        for client in self.server.clients:
            if client.active:
                topic_name = f'cam_{client.ID}/blobs'
                sub = self.create_subscription(
                    BlobArray,
                    topic_name,
                    lambda msg, client_id=client.ID: self.blob_callback(msg, client_id),
                    10
                )
                self.subscribers.append(sub)
                self.get_logger().info(f"Subscribed to client {client.ID} on topic {topic_name}")

    def blob_callback(self, msg: BlobArray, client_id: int):
        if not self.server.triangulator:
            return

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        try:
            frame_idx = int(msg.header.frame_id.split('_')[-1])
        except (ValueError, IndexError):
            self.get_logger().warn(f"Could not parse frame_idx from frame_id: {msg.header.frame_id}")
            return

        blobs_np = np.array([[blob.x_px, blob.y_px] for blob in msg.blobs], dtype=np.float32)

        if blobs_np.size > 0:
            self.server.triangulator.save(client_id, frame_idx, blobs_np)

    def publish_control_command(self, command: str):
        msg = RosString()
        msg.data = command
        self.get_logger().info(f"Publishing command: '{command}'")
        self.control_publisher.publish(msg)

def ros2_thread_func(node):
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

def format_matrix_latex(matrix):
    rows = [" & ".join(map(str, row)) for row in matrix]
    return "\\begin{bmatrix}\n" + " \\\\ \n".join(rows) + "\n\\end{bmatrix}"

def plot_calibration(server, title):
    scene = Viewer3D(title=title, size=10)
    for client in server.clients:
        scene.add_frame(client.camera.pose, client.alias, axis_size=0.4)
    scene.add_frame(np.eye(4), f"Reference", axis_size=0.4)
    return scene

# --- Initialize session state variables ---
st.set_page_config(page_title="Motion Capture Arena", layout="centered")

if "server" not in st.session_state:
    st.session_state.server = None
if "ros2_node" not in st.session_state:
    st.session_state.ros2_node = None
if "ros2_thread" not in st.session_state:
    st.session_state.ros2_thread = None
if "message_timeout" not in st.session_state:
    st.session_state.message_timeout = 2
if "wand_blobs" not in st.session_state:
    st.session_state.wand_blobs = None
if "calibration_blobs" not in st.session_state:
    st.session_state.calibration_blobs = None
if "condensed_output" not in st.session_state:
    st.session_state.condensed_output = []
if "disable_timed_capture" not in st.session_state:
    st.session_state.disable_timed_capture = True
if "online_capture_process" not in st.session_state:
    st.session_state.online_capture_process = Process(daemon=True)
if "cache_directory" not in st.session_state:
    st.session_state.cache_directory = "cache/"
    os.makedirs(st.session_state.cache_directory, exist_ok=True)

st.image("assets/mocaprasp.png")

if st.session_state.server is None:
    st.warning("Server is not initialized. Please initialize to continue.")
    if st.button("Initialize Server", use_container_width=True):
        with st.spinner("Initializing MoCap server..."):
            st.session_state.server = MoCapRasp_Server(
                config_path="config.yaml", server_address=("0.0.0.0", 25565)
            )
            # st.write(f"clients loaded: {st.session_state.server.clients}")
            # st.write(f"client addresses: {st.session_state.server.client_addresses}")
            # time.sleep(5)
        st.rerun()
else:
    transport_mode = st.radio("Select Communication Transport", ('udp', 'ros2'), horizontal=True)

    if transport_mode == 'ros2':
        if not ROS2_AVAILABLE:
            st.error("ROS2 libraries not found. Please install rclpy and mocap_interfaces.")
        elif st.session_state.ros2_node is None:
            with st.spinner("Initializing ROS2 Node..."):
                if not rclpy.ok():
                    rclpy.init()
                st.session_state.ros2_node = MocapServerNode(st.session_state.server)
                st.session_state.ros2_thread = threading.Thread(
                    target=ros2_thread_func, args=(st.session_state.ros2_node,), daemon=True
                )
                st.session_state.ros2_thread.start()
                st.success("ROS2 Node now running in the background.")
                time.sleep(1) # Warmup
    elif transport_mode == 'udp' and st.session_state.ros2_node is not None:
        if rclpy.ok():
            rclpy.shutdown()
        st.session_state.ros2_node = None
        st.session_state.ros2_thread = None
        st.info("ROS2 Node has been shutdown")

    setup_tab, calibration_tab, capture_tab = st.tabs(["⚙️ Setup", "⚖️ Calibration", "📸 Capture"])

    with setup_tab:
        st.subheader("⚙️ Arena Setup")
        st.success("Server Initialized!")

        if st.button("Register Clients", use_container_width=True):
            with st.spinner("Searching for clients on the network..."):
                registration_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                registration_sock.bind(("192.168.0.133",25555))
                registration_sock.settimeout(1.0)

                for client in st.session_state.server.clients:
                    if client.address:
                        try:
                            st.write(client.address)
                            ping_addr = tuple(client.address)

                            registration_sock.sendto(b"PING", ping_addr)

                            data, addr = registration_sock.recvfrom(1024)

                            if data == b"PONG" and addr[0] == client.address[0]:
                                client.active = True
                                client.address = addr
                                st.success(f"Received 'PONG' from Client {client.alias}")
                            else:
                                client.active = False
                                st.error(f"No message received")
                        except socket.timeout:
                            client.active = False
                            st.error("Timeout")
                        except Exception as e:
                            st.error(f"Error registering {client.alias}: {e}")
                            client.active = False
                    
                    else:
                        client.active = False
                        st.error(client.address)

                registration_sock.close()

                if transport_mode == 'ros2' and st.session_state.ros2_node:
                    st.session_state.ros2_node.discover_and_subscribe_clients()

            st.success(f"Found {len([c for c in st.session_state.server.clients if c.active])} active clients.")
            time.sleep(1)
            st.rerun()

        for client in st.session_state.server.clients:
            status_color = "green" if client.active else "red"
            status_icon = "🟢" if client.active else "🔴"
            status_text = "Online" if client.active else "Offline"

            with st.expander(label=rf"### {status_icon} :{status_color}[**{client.alias}**]"):
                st.markdown(f"**Status:** {status_text}")
                if client.address:
                    st.markdown(f"**IP Address:** `{client.address[0]}`")
                st.markdown(f"**MAC Address:** `{client.mac_address}`")
                st.markdown("**Intrinsic Matrix:**")
                st.latex(r"K = " + format_matrix_latex(client.camera.intrinsic_matrix))
                distortion_str = " & ".join(map(str, client.camera.distortion_coefficients))
                st.markdown("**Distortion Coefficients:**")
                st.latex(r"k_d = " + r"\begin{bmatrix}" + distortion_str + r"\end{bmatrix}")

    with calibration_tab:
        st.subheader("⚖️ System Calibration")

        calibration_path = "calibration/"
        os.makedirs(calibration_path, exist_ok=True)
        calibrations = sorted([
            name for name in os.listdir(calibration_path)
            if os.path.isdir(os.path.join(calibration_path, name))
        ])

        selected_folder = st.selectbox("Select a Calibration", [""] + calibrations)

        if (st.button("Load Calibration", use_container_width=True) and selected_folder != ""):
            placeholder = st.empty()
            full_path = os.path.join(calibration_path, selected_folder)
            st.session_state.server.load_calibration(full_path)
            scene = plot_calibration(server=st.session_state.server, title="Loaded Calibration")
            st.plotly_chart(scene.figure)
            placeholder.success("Calibration loaded!", icon="✅")
            time.sleep(st.session_state.message_timeout)
            placeholder.empty()
            st.session_state.server.register_clients()

        extrinsic_calibration_columns = st.columns([1, 1])
        with extrinsic_calibration_columns[0]:
            st.caption("\u200d")
            extrinsic_calibration_flag = st.button("Extrinsic Calibration", use_container_width=True)
            st.caption("AB Distance (cm)")
            AB = st.number_input("AB Distance (cm)", min_value=0.0, value=5.40, format="%0.3f", label_visibility="collapsed")
        with extrinsic_calibration_columns[1]:
            st.caption("Capture Duration (s)")
            calibration_duration = st.number_input("Extrinsic Calibration Duration (s)", min_value=15.0, value=30.0, step=15.0, format="%0.1f", label_visibility="collapsed")
            st.caption("BC Distance (cm)")
            BC = st.number_input("BC Distance (cm)", min_value=0.0, value=10.25, format="%0.3f", label_visibility="collapsed")

        if "wand_distances_calibration" not in st.session_state:
            st.session_state.wand_distances_calibration = (np.array([AB, BC, AB + BC]) * 1e-2)

        if extrinsic_calibration_flag:
            placeholder = st.empty()
            placeholder.info("Extrinsic calibration requested", icon="ℹ️")
            with st.spinner(f"Running {calibration_duration}s calibration capture..."):
                if transport_mode == 'udp':
                    if not st.session_state.server.request_sync_capture(delay_time=0.0, capture_time=calibration_duration):
                        placeholder.error("Calibration request failed!", icon="🚨")
                        time.sleep(st.session_state.message_timeout)
                        placeholder.empty()
                    else:
                        st.session_state.server.offline_capture(expected_markers=3, timeout=calibration_duration + 5)
                elif transport_mode == 'ros2' and st.session_state.ros2_node:
                    st.session_state.server.triangulator.reset()
                    st.session_state.ros2_node.publish_control_command("START")
                    time.sleep(calibration_duration)
                    st.session_state.ros2_node.publish_control_command("STOP")
                placeholder.success("Calibration request successful!", icon="✅")
                st.session_state.calibration_blobs = st.session_state.server.triangulator.full_vision()
                if not st.session_state.server.multiple_view.calibrate(st.session_state.calibration_blobs, st.session_state.wand_distances_calibration):
                    placeholder.error("Calibration failed!", icon="🚨")
                    time.sleep(st.session_state.message_timeout)
                    placeholder.empty()
                else:
                    placeholder.success("Calibration successful!", icon="✅")
                    time.sleep(st.session_state.message_timeout)
                    placeholder.empty()
                    scene = plot_calibration(server=st.session_state.server, title="Calibrated Camera Poses")
                    st.plotly_chart(scene.figure)

    with capture_tab:
        st.subheader("📸 Capture Scene")
        capture_path = os.path.join(st.session_state.cache_directory, "capture.csv")
        capture_columns = st.columns([1, 1])
        with capture_columns[1]:
            st.caption("Time Limited Capture")
            st.session_state.disable_timed_capture = not st.checkbox("Enable Timed Capture", value=False)
            st.write("")
            st.caption("Capture Delay (s)")
            capture_delay = st.number_input("Capture Delay (s)", min_value=0, step=1, label_visibility="collapsed", disabled=st.session_state.disable_timed_capture)
            if st.session_state.disable_timed_capture:
                capture_delay = 0
            st.caption("Publishing Port")
            publishing_port = st.number_input("Publishing Port", min_value=1024, max_value=65535, value=6666, step=1, label_visibility="collapsed")
            terminate_capture_flag = st.button("Terminate Capture", use_container_width=True)
        with capture_columns[0]:
            st.caption("Expected Markers")
            expected_markers = st.number_input("Marker count", min_value=0, step=1, label_visibility="collapsed")
            st.caption("Capture Duration (s)")
            capture_duration = st.number_input("Capture Duration (s)", min_value=1, step=1, label_visibility="collapsed", disabled=st.session_state.disable_timed_capture)
            if st.session_state.disable_timed_capture:
                capture_duration = -1
            st.caption("Publishing Hostname")
            publishing_hostname = st.text_input("Publishing Hostname", value=socket.gethostname(), label_visibility="collapsed")
            try:
                publishing_ip = socket.gethostbyname(publishing_hostname)
            except:
                publishing_ip = None
            start_capture_flag = st.button("Start Capture", use_container_width=True, disabled=publishing_ip is None)

        if start_capture_flag:
            placeholder = st.empty()
            if publishing_ip is None:
                placeholder.error("Publishing hostname is not valid!", icon="🚨")
                time.sleep(st.session_state.message_timeout)
                placeholder.empty()
            elif st.session_state.online_capture_process.is_alive():
                placeholder.error("Last capture is still alive! Finish it to begin new one.", icon="🚨")
                time.sleep(st.session_state.message_timeout)
                placeholder.empty()
            else:
                if transport_mode == 'udp':
                    if not st.session_state.server.request_sync_capture(delay_time=capture_delay, capture_time=capture_duration):
                        placeholder.error("UDP capture failed!")
                    else:            
                        st.session_state.online_capture_process = Process(
                            target=st.session_state.server.online_capture,
                            kwargs={
                                "expected_markers": expected_markers,
                                "visualizer_address": (publishing_ip, publishing_port),
                                "max_head": 4, "max_hold": 4, "reprojection_tol": 1,
                                "collinearity_tol": 0.01, "min_views": 3, 
                                "capture_path": capture_path, "verbose": False,
                            },
                            daemon=True,
                        )
                        st.session_state.online_capture_process.start()
                        placeholder.success("UDP capture started!")
                elif transport_mode == 'ros2' and st.session_state.ros2_node:
                    st.session_state.server.triangulator.reset()
                    placeholder.info(f"Waiting for delay: {capture_delay}s...")
                    time.sleep(capture_delay)
                    st.session_state.ros2_node.publish_control_command("START")
                    placeholder.success("ROS2 capture command sent! Clients are streaming.")

        if terminate_capture_flag:
            placeholder = st.empty()
            if transport_mode == 'udp':
                if not st.session_state.server.request_sync_capture(delay_time=0, capture_time=0):
                    placeholder.error("Termination request failed!", icon="🚨")
                    time.sleep(st.session_state.message_timeout)
                    placeholder.empty()
            elif transport_mode == 'ros2' and st.session_state.ros2_node:
                st.session_state.ros2_node.publish_control_command("STOP")
                if st.session_state.online_capture_process.is_alive():
                    st.session_state.online_capture_process.terminate()
                placeholder.success("ROS2 capture termination command sent!")

        placeholder = st.empty()
        if st.session_state.online_capture_process.is_alive():
            placeholder.success("Running capture!", icon="✅")
        while st.session_state.online_capture_process.is_alive():
            continue
        placeholder.empty()
        scene = plot_calibration(server=st.session_state.server, title="Capture Profile")
        try:
            condensed_capture_output = np.loadtxt(capture_path, delimiter=",").reshape((3, -1))
            scene.add_points(condensed_capture_output, f"Triangulated markers")
        except FileNotFoundError:
            print("[ERROR] File not found")
            pass
        except:
            print("[ERROR] Corrupted file")
            pass
        st.plotly_chart(scene.figure)

st.markdown("---")
st.markdown("Developed by [@loolirer](https://github.com/loolirer) and [@debOliveira](https://github.com/debOliveira).")