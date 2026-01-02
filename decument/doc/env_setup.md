# 环境与依赖

本章描述 Stage 1（2.1/2.2/2.3）在 Linux 下的最小运行环境、权限与常见依赖。

## 1. 系统与设备权限

### 1.1 串口（机械臂）

- 典型设备：`/dev/ttyACM0`、`/dev/ttyUSB0` 或 udev 规则映射出的 `/dev/left_arm`、`/dev/right_arm`
- 若遇到权限错误（`Permission denied`），通常需要把用户加入 `dialout` 组（不同发行版可能不同）：

```bash
sudo usermod -aG dialout $USER
```

完成后重新登录生效。

### 1.2 JoyCon（hidraw）

- JoyCon 依赖 HID 设备访问权限（`/dev/hidraw*`）。
- 若枚举不到设备或报权限问题，需要配置 udev 规则/用户组权限（不同系统策略不同）。

## 2. Python 依赖（按脚本实际 import 为准）

常见依赖（用于 SDK/IK/采集/标定）：
- `numpy`
- `scipy`
- `opencv-python`（或系统自带 OpenCV）
- `h5py`（RDT HDF5）
- `pyserial`（串口）
- `pynput`（键盘控制脚本 `arm_keyboard_control.py`）

如果你用的是 venv，可在 venv 里安装（示例）：

```bash
python -m pip install -U numpy scipy opencv-python h5py pyserial pynput
```

JoyCon 相关的 Python 依赖由 `joyconrobotics/` 代码决定（其中 HID/蓝牙依赖可能与系统环境强相关）。

## 3. Node/GitBook

- 本目录使用 GitBook（legacy）结构：`README.md` + `SUMMARY.md`。
- 生成静态站点：

```bash
gitbook build
```

输出默认在 `decument/_book/`。
