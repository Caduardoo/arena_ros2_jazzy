#include <memory>
#include <chrono>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include <opencv2/opencv.hpp>
#include "cv_bridge/cv_bridge.hpp"
#include "image_transport/image_transport.hpp"

class WebcamPublisher : public rclcpp::Node {
    public: 
    explicit WebcamPublisher() : Node("webcam_publisher_node") 
    {
        this->declare_parameter<int>("camera_index", 2);
        this->declare_parameter<double>("frame_rate",30.0);
        this->declare_parameter<std::string>("topic_name", "image_raw0");

        int camera_index = this->get_parameter("camera_index").as_int();
        double frame_rate = this->get_parameter("frame_rate").as_double();
        std::string topic_name = this->get_parameter("topic_name").as_string();

        RCLCPP_INFO(this->get_logger(), "Starting webcam index %d. framerate: %.2f Hz", camera_index, frame_rate);

        cap_.open(camera_index);

        if (!cap_.isOpened()) {
            RCLCPP_FATAL(this->get_logger(), "Camera index %d couldn't load.", camera_index);
            rclcpp::shutdown();
            return;
        }

        publisher_ = image_transport::create_publisher(this, topic_name);
        auto frame_duration = std::chrono::duration<double>(1.0 / frame_rate);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(frame_duration),
            std::bind(&WebcamPublisher::timer_callback, this)
        );
    }
    private:
    void timer_callback() {
        cv::Mat frame;
        cap_ >> frame;

        if (frame.empty()) {
            RCLCPP_WARN(this->get_logger(), "Empty frame");
            return;
        }

        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "camera_frame";
        
        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();

        publisher_.publish(std::move(msg));
    }

    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
    image_transport::Publisher publisher_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WebcamPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}