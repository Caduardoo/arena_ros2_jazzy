import rclpy
from rclpy.node import Node
import socket
import numpy as np
from mocap_interfaces.msg import Blob, BlobArray
from std_msgs.msg import Header

class UdpBridgeNode(Node):
    def __init__(self):
        super().__init__('udp_bridge_node')

        self.declare_parameter('udp_port', 25565)
        self.declare_parameter('topic_name', 'cam0/blobs')

        udp_port = self.get_parameter('udp_port').get_parameter_value().integer_value
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value

        self.publisher_ = self.create_publisher(BlobArray, self.topic_name, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_adress = ('0.0.0.0', udp_port)
        self.sock.bind(server_adress)
        self.sock.setblocking(False)

        self.get_logger().info(f"Listening to UDP port {udp_port}, publishing to '{self.topic_name}'")

        self.timer = self.create_timer(0.001, self.receive_and_publish)

    def receive_and_publish(self):
        try:
            data, adress = self.sock.recvfrom(4096) # buffer
            message = np.frombuffer(data, dtype=np.float32)
            shot_number = int(message[-2])
            timestamp = message[-1]

            blobs_np = message[:-2].reshape(-1,3)
            blob_array_msg = BlobArray()

            blob_array_msg.header = Header()
            blob_array_msg.header.stamp = self.get_clock().now().to_msg()

            cam_id_str = self.topic_name.split("/")[0]
            blob_array_msg.header.frame_id = f'{cam_id_str}_{shot_number}'

            for b in blobs_np:
                blob_msg = Blob()
                blob_msg.x_px = float(b[0])
                blob_msg.y_px = float(b[1])
                blob_msg.size = float(b[2])
                blob_array_msg.blobs.append(blob_msg)

            self.publisher_.publish(blob_array_msg)
        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"Error processing UDP packet: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = UdpBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()