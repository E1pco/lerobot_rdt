# SO-101 IK 求解器交互使用指南

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│         RViz 可视化 (robot_state_publisher)        │
│         显示机械臂模型和可视化标记                  │
└─────────────────────────────────────────────────────┘
                     ↑ /joint_states_ik
                     │
┌──────────────────────────────────────────────────────┐
│      IK 求解器节点 (ik_solver_node)                 │
│  - 订阅 /target_pose                               │
│  - 执行 LM 逆运动学求解                            │
│  - 发布 /joint_states_ik 和 /visualization_marker  │
└──────────────────────────────────────────────────────┘
         ↑ /target_pose
         │
┌──────────────────────────────────────────────────────┐
│    用户输入节点 (target_pose_input)                 │
│  - 交互式终端输入                                   │
│  - 预定义位姿库                                     │
│  - 发布 /target_pose 消息                           │
└──────────────────────────────────────────────────────┘
```

## 启动步骤

### 1. 启动完整系统（RViz + IK求解器 + 机器人发布器）

```bash
cd /home/elpco/code/lerobot/lerobot_rdt
source ros_wk/install/setup.bash
ros2 launch fishbot_description so101_ik_solver.launch.py
```

这会启动：
- `robot_state_publisher`: 发布 TF 变换
- `rviz2`: 可视化界面
- `ik_solver_node`: IK 求解器（接收目标位姿）
- `target_pose_publisher`: 演示发布器（自动发送 3 个预定义位姿）

### 2. 启动用户交互输入节点（新终端）

```bash
cd /home/elpco/code/lerobot/lerobot_rdt
source ros_wk/install/setup.bash
ros2 run fishbot_description target_pose_input
```

## 交互命令

在 `target_pose_input` 节点的终端中输入以下命令：

### help
显示完整的帮助信息

```
请输入命令: help
```

### input / i
手动输入目标位姿的笛卡尔坐标和欧拉角

```
请输入命令: input

输入末端位置 (单位: 米)
  x (m) = 0.0
  y (m) = -0.25
  z (m) = 0.25

输入欧拉角 (单位: 度)
  roll (°) = 0
  pitch (°) = 0
  yaw (°) = 0
```

### preset / p
从 7 个预定义位姿中选择

```
请输入命令: preset

可选位姿：
1. 初始位置 (Home)
2. 左前下方
3. 右前上方
4. 正上方
5. 正前方
6. 侧面水平
7. 抓取位置

请选择 (0-7): 2
```

### quit / q / exit
退出程序

```
请输入命令: quit
```

## 预定义位姿

| # | 位姿名称 | 位置 (x, y, z) m | 姿态 (R, P, Y)° | 应用 |
|---|---------|------------------|-------------------|------|
| 1 | Home | (0.0, -0.25, 0.25) | (0, 0, 0) | 初始/回中位置 |
| 2 | 左前下方 | (-0.15, -0.30, 0.10) | (45, -30, 45) | 左侧抓取 |
| 3 | 右前上方 | (0.15, -0.20, 0.30) | (-45, 30, -45) | 右侧放置 |
| 4 | 正上方 | (0.0, -0.25, 0.40) | (0, -90, 0) | 垂直放置 |
| 5 | 正前方 | (0.0, -0.40, 0.20) | (0, 0, 0) | 前方操作 |
| 6 | 侧面水平 | (0.25, -0.15, 0.25) | (0, 0, 90) | 右侧操作 |
| 7 | 抓取位置 | (0.0, -0.25, 0.05) | (0, -45, 0) | 低位抓取 |

## 工作流示例

### 场景：从 Home 位移动到左前下方再回到 Home

1. **启动系统**：
   ```bash
   # 终端1
   ros2 launch fishbot_description so101_ik_solver.launch.py
   ```

2. **启动交互输入**（在新终端中）：
   ```bash
   # 终端2
   ros2 run fishbot_description target_pose_input
   ```

3. **输入命令序列**（在终端2中）：
   ```
   请输入命令: preset
   请选择: 2      # 左前下方
   
   # 等待 IK 求解完成，RViz 中看到机械臂移动
   
   请输入命令: preset
   请选择: 1      # 返回 Home
   
   # 再次看到机械臂移动回初始位置
   ```

4. **观察 RViz**：
   - 🟢 绿色球体：当前末端执行器位置
   - 🔴 红色立方体：目标位置
   - 🟡 黄色线：位置误差向量
   - 🔵 彩色箭头：坐标系方向
   - 🔵 青色线：历史轨迹

## 键盘快捷键

在交互输入终端中：
- `Ctrl+C`: 退出程序
- `↑` / `↓`: 查看历史命令

## 输入范围参考

基于 SO-101 5DOF 机械臂的工作空间：

- **X 坐标**：-0.3 ~ 0.3 m（左右）
- **Y 坐标**：-0.4 ~ -0.1 m（前后）
- **Z 坐标**：0.0 ~ 0.4 m（上下）
- **姿态角**：±180° （任意旋转）

## 故障排除

### 问题：IK 求解失败
**原因**：目标位置可能超出工作空间或不可达
**解决方案**：
1. 检查坐标是否在工作空间范围内
2. 尝试选择预定义位姿
3. 查看 RViz 中的可视化标记

### 问题：没看到 RViz 窗口
**原因**：RViz 配置问题或显示问题
**解决方案**：
```bash
# 手动启动 RViz
source ros_wk/install/setup.bash
rviz2 -d ros_wk/install/fishbot_description/share/fishbot_description/config/so101_ik.rviz
```

### 问题：输入命令无响应
**原因**：节点可能未正确启动或主题未连接
**解决方案**：
```bash
# 检查节点状态
ros2 node list

# 检查话题
ros2 topic list

# 查看话题消息
ros2 topic echo /target_pose
```

## 性能指标

- **IK 求解时间**：～ 5-10 ms
- **位置精度**：± 3 mm
- **发布频率**：～ 100 Hz
- **末端误差监测**：实时显示在日志中

## 相关文件

- **IK 求解节点**：`fishbot_description/ik_solver_node.py`
- **用户输入节点**：`fishbot_description/target_pose_input.py`
- **机器人模型**：`../ik/robot.py` (ET 模型)
- **RViz 配置**：`config/so101_ik.rviz`
- **URDF 模型**：`../../sim/so101_new_calib.urdf`

