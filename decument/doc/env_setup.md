# 环境与依赖

## 设备权限与依赖清单

### 1) 系统与设备权限

#### 串口（机械臂）

- 典型设备：`/dev/ttyACM0`、`/dev/ttyUSB0` 或 udev 规则映射出的 `/dev/left_arm`、`/dev/right_arm`
- 若遇到权限错误（`Permission denied`），通常需要把用户加入 `dialout` 组（：

```bash
sudo usermod -aG dialout $USER
```

完成后重新登录生效。

#### JoyCon（hidraw）

- JoyCon 依赖 HID 设备访问权限（`/dev/hidraw*`）。
- 若枚举不到设备或报权限问题，需要配置 udev 规则/用户组权限（不同系统策略不同）。

### 2) Python 依赖

常见依赖（用于 SDK/IK/采集/标定）：

- `numpy`
- `scipy`
- `opencv-python`（或系统自带 OpenCV）
- `h5py`（RDT HDF5）
- `pyserial`（串口）
- `pynput`（键盘控制脚本 `arm_keyboard_control.py`）
