import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from picamera2 import Picamera2

class Picamera_publisher(Node):
    def __init__(self):
        super().__init__('Picamera_2_publisher_node')
        # PARAMS
        self.declare_parameter('topic_name', 'picamera/image_raw')
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('width', 960)
        self.declare_parameter('height', 720)
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('camera_index', 0)

        # GET PARAMS
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        frame_rate = self.get_parameter('frame_rate').get_parameter_value().double_value
        self.width_ = self.get_parameter('width').get_parameter_value().integer_value
        self.height_ = self.get_parameter('height').get_parameter_value().integer_value
        self.f_id_ = self.get_parameter('frame_id').get_parameter_value().string_value
        camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        self.resolution_ = (self.width_, self.height_)

        self.get_logger().info(f"Starting camera (index {camera_index}) with {self.width_}x{self.height_} @ {frame_rate}fps...")
        self.get_logger().info(f"Publishing raw image to {topic_name}")

        # ROS2//CV_BRIDGE

        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge_ = CvBridge()

        try:
            self.picam2_ = Picamera2(camera_index=camera_index)
            config = self.picam2_.create_video_configuration(
                main={
                    "size":self.resolution_,
                    "format":"BGR888"
                },
                controls={
                    "FrameRate": frame_rate
                }
            )
            self.picam2_.configure(config)
            self.picam2_.start()
            self.picam2_.set_controls(
                {
                    "AeEnable":False,
                    "AwbEnable":False,
                    "AnalogueGain":4.0
                }
            )
            self.get_logger().info("Camera started and controls set.")
            time.sleep(1)
        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize camera: {e}")
            self.get_logger().fatal("This node will now shut down.")
            return
        
        timer_period = 1.0 / frame_rate
        self.timer_ = self.create_timer(timer_period, self.timer_callback)
    
    def timer_callback(self):
        try:
            frame = self.picam2_.capture_array("main")
        except Exception as e:
            self.get_logger().warn(f"Failed to capture frame: {e}")
            return
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id_

        msg = self.bridge_.cv2_to_imgmsg(frame, "bgr8", header=header)

        self.publisher_.publish(msg)
    
    def destroy_node(self):
        self.get_logger().info("Shutting down camera...")
        
        if hasattr(self, 'picam2_') and self.picam2_.started:
            self.picam2_.stop()
        
        self.get_logger().info("Node shutdown...")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = Picamera_publisher()
        if hasattr(node, 'timer_'):
            rclpy.spin(node)
        else:
            node.get_logger().info("Node initialization failed, not spinning.")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Unhandled exception: {e}")
        else:
            rclpy.get_logger('main').fatal(f"Unhandled exception in init: {e}")
    finally:
        if node and rclpy.ok():
            node.destroy_node()
        rclpy.try_shutdown()
        
if __name__ == '__main__':
    main()
