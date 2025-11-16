# SO-101 机械臂 ROS2 仿真

基于 URDF 文件的 SO-101 五自由度机械臂可视化仿真环境。

## 文件结构

```
sim/
├── so101_new_calib.urdf     # 机械臂 URDF 模型文件
├── display.launch           # ROS2 launch 文件（Python格式）
├── so101.rviz              # RViz2 配置文件
├── joint_state_publisher.py # 关节状态发布器（可编程控制）
└── README.md               # 本文档
```

## 前置条件

确保已安装 ROS2 和相关工具包：

```bash
# ROS2 基础包（以 Humble 为例）
sudo apt install ros-humble-desktop

# 必需的工具包
sudo apt install ros-humble-joint-state-publisher-gui
sudo apt install ros-humble-robot-state-publisher
sudo apt install ros-humble-xacro

# Python 依赖
pip install numpy
```

## 使用方法

### 方法 1: 使用 GUI 手动控制（推荐用于测试）

启动 launch 文件，会自动打开 RViz2 和关节控制 GUI：

```bash
cd /home/elpco/code/lerobot/lerobot_rdt/sim
source /opt/ros/humble/setup.bash  # 或你的 ROS2 版本

# 直接运行 Python launch 文件
python3 display.launch
```

或使用 ros2 launch（如果已配置包）：

```bash
ros2 launch /home/elpco/code/lerobot/lerobot_rdt/sim/display.launch
```

在弹出的 **joint_state_publisher_gui** 窗口中，可以拖动滑块来控制各个关节。

### 方法 2: 使用程序化控制

在一个终端启动基础节点（不带 GUI）：

```bash
# Terminal 1: 启动 robot_state_publisher 和 RViz2
cd /home/elpco/code/lerobot/lerobot_rdt/sim
source /opt/ros/humble/setup.bash

# 修改 display.launch 中 gui 参数为 false，或手动启动节点
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$(cat so101_new_calib.urdf)" &

rviz2 -d so101.rviz &
```

在另一个终端运行关节控制脚本：

```bash
# Terminal 2: 运行关节状态发布器
cd /home/elpco/code/lerobot/lerobot_rdt/sim
source /opt/ros/humble/setup.bash

python3 joint_state_publisher.py
```

**编辑轨迹**：打开 `joint_state_publisher.py`，修改 `trajectory_mode` 或 `publish_joint_states()` 方法中的关节角度。

### 方法 3: 从 IK 求解器获取关节角度

可以结合项目中的 IK 求解器来控制机械臂：

```python
#!/usr/bin/env python3
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import sys
sys.path.insert(0, '/home/elpco/code/lerobot/lerobot_rdt')
from ik.robot import create_so101_5dof
import numpy as np

# 初始化 ROS2
rclpy.init()
node = rclpy.create_node('ik_controller')
pub = node.create_publisher(JointState, 'joint_states', 10)

# 创建机器人模型
robot = create_so101_5dof()

# 目标位姿（示例）
from scipy.spatial.transform import Rotation as R
T_goal = np.eye(4)
T_goal[:3, :3] = R.from_euler('xyz', [np.pi/4, 0, 0]).as_matrix()
T_goal[:3, 3] = [0.0, -0.25, 0.15]

# IK 求解
sol = robot.ikine_LM(Tep=T_goal, q0=np.zeros(5), mask=np.array([1,1,1,0,0,0]))

if sol.success:
    # 发布关节状态
    msg = JointState()
    msg.header = Header()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = robot.joint_names
    msg.position = sol.q.tolist()
    
    pub.publish(msg)
    print(f"Published joint positions: {sol.q}")
else:
    print(f"IK failed: {sol.reason}")

node.destroy_node()
rclpy.shutdown()
```

## RViz2 配置说明

`so101.rviz` 配置文件已预设以下显示：

- **Grid**: 网格参考平面
- **RobotModel**: 机械臂 3D 模型（从 `/robot_description` 话题读取）
- **TF**: 坐标系变换关系树

可以在 RViz2 中添加更多显示项：
- **Axes**: 显示坐标轴
- **Path**: 显示末端轨迹
- **MarkerArray**: 显示目标位置等

## 关节说明

URDF 中定义的 5 个关节：

1. `shoulder_pan` - 肩部旋转（Rz）
2. `shoulder_lift` - 肩部抬升（Ry）
3. `elbow_flex` - 肘部弯曲（Ry）
4. `wrist_flex` - 腕部俯仰（Ry）
5. `wrist_roll` - 腕部滚转（Rx）

关节限位（rad）：
```
[-1.92, -1.75, -1.69, -1.66, -2.74]  # 下限
[ 1.92,  1.75,  1.69,  1.66,  2.84]  # 上限
```

## 故障排除

### 问题 1: 找不到 URDF 文件

确保在 `sim/` 目录下运行，或修改 `display.launch` 中的路径为绝对路径。

### 问题 2: RViz2 无法显示模型

检查：
- `/robot_description` 话题是否有数据：`ros2 topic echo /robot_description`
- `robot_state_publisher` 节点是否运行：`ros2 node list`
- TF 树是否正确：`ros2 run tf2_tools view_frames`

### 问题 3: mesh 文件路径错误

URDF 中引用了 `assets/` 目录下的 STL 文件。如果显示异常，确保：
- `assets/` 目录与 URDF 文件在同一目录
- 或修改 URDF 中的 `filename` 为绝对路径

### 问题 4: 关节不动

确保 `joint_state_publisher` 或 `joint_state_publisher_gui` 正在运行并发布到 `/joint_states` 话题：

```bash
ros2 topic echo /joint_states
```

## 扩展功能

### 添加轨迹记录

```python
import csv

positions_history = []

def record_callback():
    # 在发布回调中添加
    positions_history.append(msg.position)
    
# 保存到 CSV
with open('trajectory.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'])
    writer.writerows(positions_history)
```

### 集成 MoveIt2

如果需要运动规划功能，可以配置 MoveIt2：

```bash
sudo apt install ros-humble-moveit
# 然后创建 MoveIt 配置包（需要额外配置）
```

## 相关命令

```bash
# 查看所有话题
ros2 topic list

# 查看关节状态
ros2 topic echo /joint_states

# 查看 TF 树
ros2 run tf2_tools view_frames

# 查看节点
ros2 node list

# 手动发布关节状态（测试用）
ros2 topic pub /joint_states sensor_msgs/msg/JointState \
  "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, \
    name: ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'], \
    position: [0.0, -0.5, 0.5, 0.0, 0.0]}"
```

## 参考资料

- [ROS2 URDF Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)
- [RViz2 User Guide](https://github.com/ros2/rviz)
- [robot_state_publisher](https://github.com/ros/robot_state_publisher)
