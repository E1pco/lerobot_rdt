#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class URDFPublisher(Node):
    def __init__(self):
        super().__init__('urdf_pub')
        self.pub = self.create_publisher(String, '/robot_description', 10)

        urdf = open('so101_new_calib.urdf').read()
        msg = String()
        msg.data = urdf
        self.pub.publish(msg)

        self.get_logger().info("Published URDF to /robot_description")

def main():
    rclpy.init()
    node = URDFPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
