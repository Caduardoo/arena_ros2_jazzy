import rclpy
from rclpy.node import Node

from mocap_interfaces.msg import Target
from geometry_msgs.msg import Point

class target_detector_node(Node):
    def __init__(self):
        super().__init__('Target_Detector')
        # Type, Topic_name, Queue Size
        self.publisher_=self.create_publisher(Target,'Target_position',10)

        timer_period = 2.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Target detector node Started. Publishing...')

    def timer_callback(self):
        msg = Target()

        # TO DO: Add 2D marker detection Logic

        msg.target_name = "test_target"
        msg.position = Point(x=1.0,y=2.0,z=0.0)

        self.publisher_.publish(msg)

        self.get_logger().info(f'Publishing: "{msg.target_name}" at position (x={msg.position.x}, y={msg.position.y})')
    
def main():
    rclpy.init()

    node = target_detector_node()

    rclpy.spin(node)

    rclpy.shutdown()

if __name__=='__main__':
    main()
