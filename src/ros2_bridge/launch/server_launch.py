from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    clients = [
        {'id': 0, 'port': 25566, 'topic': 'cam_0/blobs'},
        {'id': 1, 'port': 25567, 'topic': 'cam_1/blobs'},
        {'id': 2, 'port': 25568, 'topic': 'cam_2/blobs'},
        {'id': 3, 'port': 25569, 'topic': 'cam_3/blobs'},
        
    ]

    bridge_nodes = [
        Node(
            package='ros2_bridge',
            executable='udp_bridge_node',
            name=f'udp_bridge_{client["id"]}', # Give each node a unique name
            parameters=[
                {'udp_port': client['port']},
                {'topic_name': client['topic']}
            ]
        ) for client in clients
    ]

    # Start the Streamlit GUI
    streamlit_process = ExecuteProcess(
        cmd=['python3', '-m', 'streamlit', 'run', 'src/ros2_bridge/mocaprasp_gui_ros copy.py'],
        output='screen'
    )

    launch_actions = bridge_nodes + [streamlit_process]

    return LaunchDescription(launch_actions)