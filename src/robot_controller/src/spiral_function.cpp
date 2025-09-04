#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <chrono>

class spiral_function_node : public rclcpp::Node {
    public:
    spiral_function_node() : Node("spiral_function_node") {
        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel",10);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&spiral_function_node::timer_callback, this));

        RCLCPP_INFO(this->get_logger(), "Spiral controller node: Starting");
    }
    private: 
    void timer_callback() {
        auto msg = geometry_msgs::msg::Twist();
        msg.linear.x = linear_velocity_x_;
        msg.angular.z = angular_velocity_z_;
        publisher_->publish(msg);
        angular_velocity_z_ += angular_velocity_increment_;
        
        if (angular_velocity_z_ > max_angular_velocity_) {
            angular_velocity_z_ = max_angular_velocity_;
        }
    }

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    double linear_velocity_x_ = 0.4;
    double angular_velocity_z_ = 0.0;
    double angular_velocity_increment_ = 0.02;
    double max_angular_velocity_ = 3.0;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<spiral_function_node>());
    rclcpp::shutdown();
    return 0;
} 