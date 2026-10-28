"""ROS 2 camera publisher for repeatable image-stream emulation."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import SensorDataQoS
from sensor_msgs.msg import Image

from ab.bot.runtime import synthetic_rgb_frame


class SyntheticCameraNode(Node):
    def __init__(self):
        super().__init__("nn_bot_synthetic_camera")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        if not math.isfinite(self.publish_rate_hz) or self.publish_rate_hz <= 0:
            raise ValueError("publish_rate_hz must be finite and positive")
        if self.width < 1 or self.height < 1:
            raise ValueError(f"Frame dimensions must be positive, got {self.width}x{self.height}")
        self.frame_index = 0
        self.publisher = self.create_publisher(
            Image, self.image_topic, SensorDataQoS()
        )
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.publish_frame)
        self.get_logger().info(
            f"Publishing {self.width}x{self.height} RGB frames to {self.image_topic} "
            f"at {self.publish_rate_hz:g} Hz"
        )

    def publish_frame(self):
        frame = synthetic_rgb_frame(self.width, self.height, self.frame_index)
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "synthetic_camera"
        message.height = self.height
        message.width = self.width
        message.encoding = "rgb8"
        message.is_bigendian = 0
        message.step = self.width * 3
        message.data = frame.tobytes()
        self.publisher.publish(message)
        self.frame_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SyntheticCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
