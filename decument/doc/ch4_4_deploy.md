# 4.4 RDT 模型部署与推理

本节说明如何将微调后的 RDT 模型部署到真实机器人上进行推理。

---

## 1. 部署架构

```
语言指令 ──→ T5 编码器 ──→ 语言嵌入
                              ↓
三视角图像 ──→ SigLIP 编码器 ──→ 视觉嵌入 ──→ RDT ──→ 64步动作序列
                              ↓                        ↓
当前状态 ────────────────→ 状态嵌入 ──┘          动作执行器
```

---

## 2. 推理脚本配置

### 2.1 基础推理脚本（inference.sh）

```bash
#!/bin/bash
# inference.sh - SO-101 双臂推理配置

python -m scripts.so101_inference \
    --use_actions_interpolation \
    --pretrained_model_name_or_path="checkpoints/rdt-finetune-so101/checkpoint-150000" \
    --lang_embeddings_path="outs/lang_embeddings/pick_and_place.pt" \
    --ctrl_freq=30
```

### 2.2 参数说明

| 参数 | 说明 |
|-----|------|
| `pretrained_model_name_or_path` | 微调后检查点路径 |
| `lang_embeddings_path` | 预计算的语言嵌入文件 |
| `ctrl_freq` | 控制频率（与采集时一致） |
| `use_actions_interpolation` | 动作插值（更平滑） |

---

## 3. 推理代码实现

### 3.1 模型加载类

创建 `scripts/so101_model.py`：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 双臂 RDT 推理模型封装"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.rdt_runner import RDTRunner
from configs.state_vec import STATE_VEC_IDX_MAPPING


class SO101RDTModel:
    """SO-101 双臂 RDT 推理封装类"""
    
    def __init__(
        self,
        pretrained_path: str,
        lang_embeddings_path: str = None,
        device: str = "cuda",
        ctrl_freq: int = 30,
    ):
        """
        初始化推理模型
        
        Args:
            pretrained_path: 微调后检查点路径
            lang_embeddings_path: 预计算语言嵌入路径
            device: 推理设备
            ctrl_freq: 控制频率
        """
        self.device = device
        self.ctrl_freq = ctrl_freq
        
        # 加载模型
        print(f"Loading model from {pretrained_path}...")
        self.model = RDTRunner.from_pretrained(pretrained_path)
        self.model.to(device)
        self.model.eval()
        
        # 加载语言嵌入
        if lang_embeddings_path and os.path.exists(lang_embeddings_path):
            self.lang_embeddings = torch.load(lang_embeddings_path, map_location=device)
        else:
            self.lang_embeddings = None
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 历史缓存
        self.image_history = []
        self.action_buffer = []
        self.action_idx = 0
        
        print("Model loaded successfully!")
    
    def reset(self):
        """重置推理状态"""
        self.image_history = []
        self.action_buffer = []
        self.action_idx = 0
    
    def _format_state(self, left_joints, left_gripper, right_joints, right_gripper):
        """
        将 SO-101 状态转换为统一向量
        
        Args:
            left_joints: [6] 左臂关节角度 (rad)
            left_gripper: float 左夹爪 (0-1)
            right_joints: [6] 右臂关节角度 (rad)
            right_gripper: float 右夹爪 (0-1)
        
        Returns:
            state: [128] 统一状态向量
        """
        state = np.zeros(128, dtype=np.float32)
        
        # 右臂
        state[0:6] = right_joints
        state[10] = right_gripper
        
        # 左臂
        state[50:56] = left_joints
        state[60] = left_gripper
        
        return state
    
    def _unformat_action(self, action):
        """
        从统一向量提取 SO-101 动作
        
        Args:
            action: [128] 统一动作向量
        
        Returns:
            left_joints, left_gripper, right_joints, right_gripper
        """
        right_joints = action[0:6]
        right_gripper = action[10]
        left_joints = action[50:56]
        left_gripper = action[60]
        
        return left_joints, left_gripper, right_joints, right_gripper
    
    def _preprocess_images(self, images_dict):
        """
        预处理图像
        
        Args:
            images_dict: {
                'cam_exterior': np.ndarray (H, W, 3),
                'cam_right_wrist': np.ndarray (H, W, 3),
                'cam_left_wrist': np.ndarray (H, W, 3)  # 可选
            }
        
        Returns:
            images_tensor: [6, 3, 224, 224] (t-1 和 t 时刻各 3 视角)
        """
        current_images = []
        
        for cam_name in ['cam_exterior', 'cam_right_wrist', 'cam_left_wrist']:
            if cam_name in images_dict:
                img = Image.fromarray(images_dict[cam_name])
                img_tensor = self.image_transform(img)
            else:
                # 缺失视角用零填充
                img_tensor = torch.zeros(3, 224, 224)
            current_images.append(img_tensor)
        
        current_images = torch.stack(current_images)  # [3, 3, 224, 224]
        
        # 维护历史
        if len(self.image_history) == 0:
            # 首帧，t-1 用当前帧代替
            self.image_history = current_images.clone()
        
        # 拼接 t-1 和 t 时刻
        images_tensor = torch.cat([self.image_history, current_images], dim=0)  # [6, 3, 224, 224]
        
        # 更新历史
        self.image_history = current_images.clone()
        
        return images_tensor.unsqueeze(0).to(self.device)  # [1, 6, 3, 224, 224]
    
    def step(self, images_dict, left_joints, left_gripper, right_joints, right_gripper):
        """
        执行一步推理
        
        Args:
            images_dict: 图像字典
            left_joints: [6] 左臂关节角度
            left_gripper: 左夹爪开合
            right_joints: [6] 右臂关节角度
            right_gripper: 右夹爪开合
        
        Returns:
            left_joints_cmd, left_gripper_cmd, right_joints_cmd, right_gripper_cmd
        """
        # 如果动作缓冲区有数据，直接返回下一个动作
        if self.action_idx < len(self.action_buffer):
            action = self.action_buffer[self.action_idx]
            self.action_idx += 1
            return self._unformat_action(action)
        
        # 需要重新推理
        with torch.no_grad():
            # 准备输入
            images = self._preprocess_images(images_dict)
            state = self._format_state(left_joints, left_gripper, right_joints, right_gripper)
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # [1, 128]
            
            # 推理
            actions = self.model.predict_action(
                images=images,
                state=state_tensor,
                lang_embeddings=self.lang_embeddings,
            )  # [1, 64, 128]
            
            # 更新动作缓冲区
            self.action_buffer = actions[0].cpu().numpy()  # [64, 128]
            self.action_idx = 1
            
            return self._unformat_action(self.action_buffer[0])
```

### 3.2 主推理循环

创建 `scripts/so101_inference.py`：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-101 双臂 RDT 实机推理"""

import sys
import time
import argparse
import numpy as np

# 添加项目路径
sys.path.insert(0, '/path/to/lerobot_rdt')

from scripts.so101_model import SO101RDTModel
from driver.ftservo_controller import ServoController


class SO101RDTInference:
    """SO-101 双臂 RDT 推理控制器"""
    
    def __init__(
        self,
        pretrained_path: str,
        lang_embeddings_path: str,
        left_port: str = "/dev/left_arm",
        right_port: str = "/dev/right_arm",
        left_config: str = "./driver/left_arm.json",
        right_config: str = "./driver/right_arm.json",
        ctrl_freq: int = 30,
        use_interpolation: bool = True,
    ):
        print("=" * 60)
        print("SO-101 RDT Inference Controller")
        print("=" * 60)
        
        # 初始化舵机控制器
        print("\n[1/3] Connecting servo controllers...")
        self.left_controller = ServoController(
            port=left_port, baudrate=1_000_000, config_path=left_config
        )
        self.right_controller = ServoController(
            port=right_port, baudrate=1_000_000, config_path=right_config
        )
        print("✓ Servo controllers connected")
        
        # 初始化相机（示例）
        print("\n[2/3] Initializing cameras...")
        self.cameras = self._init_cameras()
        print("✓ Cameras initialized")
        
        # 加载 RDT 模型
        print("\n[3/3] Loading RDT model...")
        self.model = SO101RDTModel(
            pretrained_path=pretrained_path,
            lang_embeddings_path=lang_embeddings_path,
            ctrl_freq=ctrl_freq,
        )
        print("✓ Model loaded")
        
        self.ctrl_freq = ctrl_freq
        self.use_interpolation = use_interpolation
        self.running = True
    
    def _init_cameras(self):
        """初始化相机（根据实际硬件修改）"""
        import cv2
        cameras = {}
        # 外部相机
        cameras['cam_exterior'] = cv2.VideoCapture(2)
        # 右腕相机
        cameras['cam_right_wrist'] = cv2.VideoCapture(4)
        # 左腕相机
        cameras['cam_left_wrist'] = cv2.VideoCapture(0)
        return cameras
    
    def _capture_images(self):
        """采集图像"""
        images = {}
        for name, cap in self.cameras.items():
            ret, frame = cap.read()
            if ret:
                images[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                images[name] = np.zeros((480, 640, 3), dtype=np.uint8)
        return images
    
    def _read_state(self):
        """读取当前状态"""
        # 读取关节角度（示例，需要根据实际 API 修改）
        left_joints = np.zeros(6)
        right_joints = np.zeros(6)
        left_gripper = 0.5
        right_gripper = 0.5
        
        # TODO: 从 controller 读取实际关节角度
        
        return left_joints, left_gripper, right_joints, right_gripper
    
    def _execute_action(self, left_joints, left_gripper, right_joints, right_gripper):
        """执行动作"""
        # TODO: 将关节角度转换为舵机命令并执行
        pass
    
    def run(self):
        """主推理循环"""
        print("\n" + "=" * 60)
        print("🤖 RDT Inference Running")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
        
        self.model.reset()
        dt = 1.0 / self.ctrl_freq
        
        try:
            while self.running:
                t_start = time.time()
                
                # 采集图像
                images = self._capture_images()
                
                # 读取状态
                left_joints, left_gripper, right_joints, right_gripper = self._read_state()
                
                # 推理
                cmd = self.model.step(
                    images,
                    left_joints, left_gripper,
                    right_joints, right_gripper
                )
                
                # 执行
                self._execute_action(*cmd)
                
                # 控制频率
                elapsed = time.time() - t_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n⛔ Stopped by user")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        for cap in self.cameras.values():
            cap.release()
        print("Resources released")


def main():
    parser = argparse.ArgumentParser(description="SO-101 RDT Inference")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--lang_embeddings_path", type=str, required=True)
    parser.add_argument("--ctrl_freq", type=int, default=30)
    parser.add_argument("--use_actions_interpolation", action="store_true")
    
    args = parser.parse_args()
    
    controller = SO101RDTInference(
        pretrained_path=args.pretrained_model_name_or_path,
        lang_embeddings_path=args.lang_embeddings_path,
        ctrl_freq=args.ctrl_freq,
        use_interpolation=args.use_actions_interpolation,
    )
    controller.run()


if __name__ == "__main__":
    main()
```

---

## 4. 动作插值

为使动作更平滑，可以启用动作插值：

```python
def interpolate_actions(self, prev_action, curr_action, num_steps=5):
    """
    动作插值
    
    Args:
        prev_action: 上一步动作 [128]
        curr_action: 当前动作 [128]
        num_steps: 插值步数
    
    Returns:
        interpolated: [num_steps, 128]
    """
    alphas = np.linspace(0, 1, num_steps + 1)[1:]
    interpolated = []
    for alpha in alphas:
        action = (1 - alpha) * prev_action + alpha * curr_action
        interpolated.append(action)
    return np.array(interpolated)
```

---

## 5. 安全与限位

### 5.1 关节限位

```python
def clamp_joints(self, joints, joint_limits):
    """
    关节限位
    
    Args:
        joints: [6] 关节角度
        joint_limits: [(min, max), ...] 每个关节的限位
    
    Returns:
        clamped: [6] 限位后的关节角度
    """
    clamped = joints.copy()
    for i, (jmin, jmax) in enumerate(joint_limits):
        clamped[i] = np.clip(clamped[i], jmin, jmax)
    return clamped
```

### 5.2 速度限制

```python
def limit_velocity(self, prev_joints, curr_joints, max_vel, dt):
    """
    速度限制
    
    Args:
        prev_joints: 上一时刻关节角度
        curr_joints: 当前目标关节角度
        max_vel: 最大角速度 (rad/s)
        dt: 时间间隔
    
    Returns:
        limited: 速度限制后的目标
    """
    delta = curr_joints - prev_joints
    max_delta = max_vel * dt
    delta = np.clip(delta, -max_delta, max_delta)
    return prev_joints + delta
```

---

## 6. 推理优化

### 6.1 半精度推理

```python
# 模型转半精度
self.model = self.model.half()

# 输入也需要转半精度
images = images.half()
state = state.half()
```

### 6.2 TensorRT 加速

```python
import torch_tensorrt

# 编译模型
trt_model = torch_tensorrt.compile(
    self.model,
    inputs=[
        torch_tensorrt.Input(shape=[1, 6, 3, 224, 224], dtype=torch.half),
        torch_tensorrt.Input(shape=[1, 128], dtype=torch.half),
    ],
    enabled_precisions={torch.half}
)
```

---

## 7. 调试与可视化

### 7.1 动作可视化

```python
import matplotlib.pyplot as plt

def visualize_action_chunk(actions):
    """可视化动作序列"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 右臂关节
    for i in range(6):
        axes[0].plot(actions[:, i], label=f'right_joint_{i}')
    axes[0].legend()
    axes[0].set_title('Right Arm Joints')
    
    # 左臂关节
    for i in range(6):
        axes[1].plot(actions[:, 50+i], label=f'left_joint_{i}')
    axes[1].legend()
    axes[1].set_title('Left Arm Joints')
    
    plt.tight_layout()
    plt.savefig('action_chunk.png')
```

### 7.2 实时监控

```python
def log_step(self, step, images, state, action):
    """记录推理步骤"""
    print(f"Step {step}:")
    print(f"  State: {state[:6]} (right arm)")
    print(f"  Action: {action[:6]} (right arm)")
```

---

## 8. 常见问题

### Q1: 推理延迟太高

**解决方案**：
1. 使用半精度 (`model.half()`)
2. 减少图像分辨率
3. 使用 TensorRT 加速
4. 减小 action chunk 使用频率

### Q2: 动作抖动

**解决方案**：
1. 启用动作插值
2. 增加低通滤波
3. 检查控制频率是否与训练一致

### Q3: 任务执行失败

**排查步骤**：
1. 检查语言指令是否正确
2. 确认相机视角与训练一致
3. 验证状态向量映射是否正确
4. 检查关节限位是否合理
