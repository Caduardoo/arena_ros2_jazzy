import rclpy
from rclpy.node import Node

from mocap_interfaces.msg import Target

class target_subscriber_node(Node):
    def __init__(self):
        super().__init__('Target_Subscriber')

        # Target, Topic_name, Callback: self.listener_callback, Queue size
        self.subscription = self.create_subscription(
            Target,
            'Target_position',
            self.listener_callback,
            10
        )
        self.get_logger().info('Target Subscriber started. Waiting for messages...')

    def listener_callback(self, msg):
        # TO DO: add message processing logic
        self.get_logger().info(f'Received:"{msg.target_name}" at position: (x={msg.position.x}, y={msg.position.y})')

def main(args=None):
    rclpy.init(args=args)
    
    node = target_subscriber_node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()