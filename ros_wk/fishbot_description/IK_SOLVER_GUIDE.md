# SO-101 机械臂 IK 求解可视化指南

在 ROS2 工作空间中运行逆解计算并在 RViz 中进行可视化。

## 快速开始

### 1. 编译包

```bash
cd ~/code/lerobot/lerobot_rdt/ros_wk
colcon build --packages-select fishbot_description
source install/setup.bash
```

### 2. 启动完整系统

```bash
ros2 launch fishbot_description so101_ik_solver.launch.py
```

这将启动以下节点：
- `robot_state_publisher`: 发布机器人模型 TF
- `rviz2`: RViz 可视化工具
- `joint_state_publisher_gui`: 关节控制 GUI（可选）
- `so101_ik_solver`: IK 求解器节点
- `target_pose_publisher`: 目标位姿演示发布器

### 3. 观察结果

在 RViz 中您将看到：
- **绿色球体**: 当前末端执行器位置（根据 IK 求解结果）
- **红色立方体**: 目标位置
- **黄色连接线**: 从末端到目标的距离
- **彩色箭头**: 末端坐标系（X 红、Y 绿、Z 蓝）
- **青色轨迹**: 末端运动历史轨迹

## 手动发送目标位姿

除了使用演示发布器，您也可以手动发送目标位姿：

```bash
ros2 topic pub /target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'base_link'}, \
    pose: {position: {x: 0.0, y: -0.3, z: 0.15}, \
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"
```

### 示例位姿

**位置 1 - 前方**
```bash
ros2 topic pub /target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'base_link'}, \
    pose: {position: {x: 0.1, y: -0.25, z: 0.20}, \
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"
```

**位置 2 - 侧方**
```bash
ros2 topic pub /target_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'base_link'}, \
    pose: {position: {x: -0.1, y: -0.30, z: 0.10}, \
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"
```

## 实时监控关节状态

在另一个终端查看 IK 求解器发布的关节状态：

```bash
ros2 topic echo /joint_states_ik
```

您将看到：
```
header:
  stamp:
    sec: 1234567890
    nanosec: 123456789
  frame_id: base_link
name:
- shoulder_pan
- shoulder_lift
- elbow_flex
- wrist_flex
- wrist_roll
position:
- 0.123
- -0.456
- 0.789
- 0.234
- -0.567
velocity: []
effort: []
```

## 查看可视化标记信息

```bash
ros2 topic echo /visualization_marker_array
```

## 调试和故障排除

### 检查节点是否运行

```bash
ros2 node list
```

应该看到：
```
/rviz2
/robot_state_publisher
/so101_ik_solver
/target_pose_publisher
/joint_state_publisher_gui
```

### 检查话题是否发布

```bash
ros2 topic list
```

关键话题：
- `/joint_states_ik`: IK 求解器发布的关节状态
- `/target_pose`: 接收的目标位姿
- `/visualization_marker_array`: 可视化标记
- `/robot_description`: 机器人 URDF 模型

### 查看 IK 求解器日志

```bash
ros2 node info /so101_ik_solver
```

或在启动 launch 文件的终端中查看输出。

### 常见错误

**错误 1: "package 'fishbot_description' not found"**
```bash
# 确保已编译包
cd ~/code/lerobot/lerobot_rdt/ros_wk
colcon build --packages-select fishbot_description
source install/setup.bash
```

**错误 2: IK 求解失败 "IK failed"**
- 检查目标位置是否在机械臂工作空间内
- 尝试调整初始关节角度（q0）
- 尝试使用不同的 IK 求解方法

**错误 3: RViz 无法显示机器人**
- 确保 `/robot_description` 话题有数据
- 检查 `robot_state_publisher` 是否运行
- 在 RViz 中设置 Fixed Frame 为 `base_link`

## 修改演示轨迹

编辑 `target_pose_publisher.py` 中的 `self.targets` 列表：

```python
self.targets = [
    {
        'name': '目标 1',
        'x': 0.0, 'y': -0.25, 'z': 0.15,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        'duration': 5  # 秒
    },
    # ... 添加更多目标位姿
]
```

然后重新编译并运行：

```bash
cd ~/code/lerobot/lerobot_rdt/ros_wk
colcon build --packages-select fishbot_description
source install/setup.bash
ros2 launch fishbot_description so101_ik_solver.launch.py
```

## 高级用法

### 集成到自己的 ROS2 节点

```python
import rclpy
from geometry_msgs.msg import PoseStamped, Point, Quaternion

def main():
    rclpy.init()
    node = rclpy.create_node('my_ik_controller')
    pub = node.create_publisher(PoseStamped, 'target_pose', 10)
    
    # 发送目标位姿
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'
    msg.pose.position = Point(x=0.0, y=-0.25, z=0.15)
    msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    
    pub.publish(msg)
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

### 保存和回放轨迹

使用 ROS2 bag 记录和回放：

```bash
# 记录
ros2 bag record /target_pose /joint_states_ik -o my_trajectory

# 回放
ros2 bag play my_trajectory
```

## 参考文档

- [ROS2 文档](https://docs.ros.org/en/humble/)
- [RViz2 用户指南](https://github.com/ros2/rviz)
- [Geometry Msgs](http://docs.ros.org/en/humble/p/geometry_msgs/)
- [Visualization Msgs](http://docs.ros.org/en/humble/p/visualization_msgs/)
