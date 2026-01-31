# 主从臂遥操作 (Leader-Follower)

**脚本位置**：`driver/teleop_leader_follower.py`

## 简介
该模块实现了基于串口总线舵机的主从臂实时遥操作功能。系统通过读取“主臂”（Leader）的关节角度，经过归一化和重映射处理，驱动“从臂”（Follower）跟随运动。

主要特性包括：
1. **异构映射支持**：主从臂可以是不同的机械结构。只要通过 JSON 配置文件定义好每个关节的有效活动范围（`range_min`, `range_max`），程序会自动将主臂的运动比例（0%~100%）映射到从臂的对应范围内。
2. **多圈/跨零处理**：通过最短路径算法（Shortest Delta）和连续空间展开（Unwrap），完美处理舵机 0-4096 (0°-360°) 的边界跨越问题，避免死锁或突变。
3. **低通滤波平滑**：内置一阶低通滤波器，通过 `--alpha` 参数调节响应灵敏度与平滑度，有效消除手部操作的微小抖动。

## 快速开始

在 `driver` 目录下运行：

```bash
# 启动主从遥操
python teleop_leader_follower.py \
  --leader-port /dev/right_leader \
  --follower-port /dev/right_arm \
  --leader-config right_arm_leader.json \
  --follower-config right_arm.json \
  --speed 1200
```

## 参数详解

使用 `python teleop_leader_follower.py --help` 可查看完整参数。

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--leader-port` | `/dev/right_leader` | 主臂（遥控端）的串口端口号 |
| `--follower-port` | `/dev/right_arm` | 从臂（执行端）的串口端口号 |
| `--leader-config` | `right_arm_leader.json` | 主臂配置文件，需包含关节 ID 及活动范围定义 |
| `--follower-config` | `right_arm.json` | 从臂配置文件 |
| `--baudrate` | `1000000` | 舵机通信波特率 |
| `--rate` | `30.0` | 控制循环频率 (Hz)，建议 30-60Hz |
| `--speed` | `1200` | 从臂舵机的移动速度限制（单位：步/秒），越小越安全 |
| `--alpha` | `0.35` | 平滑系数 (0.0 ~ 1.0)。<br>值越小：滤波越强，动作越平滑，但延迟增加。<br>值越大：跟随越紧密，但可能引入抖动。 |
| `--joints` | (自动匹配) | 指定需要同步的关节名称（逗号分隔），如 `joint1,joint2`。默认自动匹配主从配置中名称相同的关节。 |
| `--debug` | (关闭) | 开启后会在终端打印详细的映射数值和调试信息。 |

## 原理说明

### 1. 归一化映射 (Normalization & Mapping)
程序并不会直接生硬地将主臂的原始 0-4095 坐标值发给从臂，而是基于**相对行程**进行映射。

对于任意关节 $i$：
1. 读取主臂位置 $Pos_{leader}$。
2. 根据主臂配置的 `[range_min, range_max]`，计算当前位置处于该扇区的进度 $Progress \in [0, 1]$。
3. 读取从臂配置的 `[range_min, range_max]`，计算该进度对应的目标位置 $Target_{follower}$。

$$ Target = Min_{follower} + Progress \times (Max_{follower} - Min_{follower}) $$

这使得你可以用一个活动范围较小的主臂（如手柄），控制一个活动范围较大的从臂，只要双方覆盖的逻辑动作一致即可。

### 2. 空间展开 (Unwrapping)
舵机是 12 位精度（0-4095）。为了防止在 0 和 4095 交界处发生数值跳变（例如从 4090 变为 10，数值看似剧变，实际物理上只动了一点点），程序内部维护了一个**展开的连续坐标系**。所有的滤波和差值计算都在这个连续空间进行，最后下发指令时再转换回舵机指令。
