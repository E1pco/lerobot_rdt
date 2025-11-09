#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四连杆机械臂姿态演示 (Roll / Pitch / Yaw)
Author: Elpco Han / ChatGPT-5
Date: 2025-11-09
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

# ----------------------------------------
# 机械臂几何参数 (单位: 米)
# ----------------------------------------
L1, L2, L3 = 0.08, 0.10, 0.06  # 基座到肩, 肩到肘, 肘到腕
EE_LENGTH = 0.04               # 末端长度

# ----------------------------------------
# 绘制坐标轴
# ----------------------------------------
def draw_axes(ax, origin, R_mat, length=0.03, label=""):
    colors = ['r', 'g', 'b']
    for i in range(3):
        vec = R_mat[:, i] * length
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2], color=colors[i], linewidth=2)
    if label:
        ax.text(*origin, label, color="k", fontsize=10)

# ----------------------------------------
# 前向运动学 (简化版 3 连杆)
# ----------------------------------------
def forward_kinematics(theta1, theta2, theta3):
    """
    theta1: 基座旋转 (绕Z)
    theta2: 肩部俯仰 (绕Y)
    theta3: 肘部俯仰 (绕Y)
    返回: 各关键点位置 (base, shoulder, elbow, wrist)
    """
    base = np.array([0, 0, 0])
    shoulder = np.array([0, 0, L1])

    Rz1 = R.from_euler('z', theta1).as_matrix()
    Ry2 = R.from_euler('y', theta2).as_matrix()
    Ry3 = R.from_euler('y', theta3).as_matrix()

    elbow = shoulder + Rz1 @ Ry2 @ np.array([0, 0, L2])
    wrist = elbow + Rz1 @ Ry2 @ Ry3 @ np.array([0, 0, L3])

    return base, shoulder, elbow, wrist

# ----------------------------------------
# 动画更新函数
# ----------------------------------------
def update(frame):
    ax.cla()
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([0, 0.25])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("机械臂末端姿态演示 (Roll / Pitch / Yaw)")

    # 基座 → 肩 → 肘 → 腕
    theta1 = np.deg2rad(0)
    theta2 = np.deg2rad(45)
    theta3 = np.deg2rad(-30)
    base, shoulder, elbow, wrist = forward_kinematics(theta1, theta2, theta3)

    # Roll / Pitch / Yaw 动态变化
    roll = np.deg2rad(min(frame, 60))       # 0→60°
    pitch = np.deg2rad(min(max(frame - 60, 0), 60))
    yaw = np.deg2rad(min(max(frame - 120, 0), 60))
    R_ee = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

    # 末端点 (EE)
    end_effector = wrist + R_ee @ np.array([0, 0, EE_LENGTH])

    # 绘制连杆
    arm_x = [base[0], shoulder[0], elbow[0], wrist[0], end_effector[0]]
    arm_y = [base[1], shoulder[1], elbow[1], wrist[1], end_effector[1]]
    arm_z = [base[2], shoulder[2], elbow[2], wrist[2], end_effector[2]]
    ax.plot(arm_x, arm_y, arm_z, '-o', color='k', linewidth=3, markersize=5)

    # 绘制坐标轴
    draw_axes(ax, wrist, R_ee, length=0.04, label="EE")

    # 显示角度数值
    ax.text2D(0.05, 0.92, f"Roll  = {np.rad2deg(roll):6.2f}°", transform=ax.transAxes, color='r')
    ax.text2D(0.05, 0.88, f"Pitch = {np.rad2deg(pitch):6.2f}°", transform=ax.transAxes, color='g')
    ax.text2D(0.05, 0.84, f"Yaw   = {np.rad2deg(yaw):6.2f}°", transform=ax.transAxes, color='b')

    # 阶段提示
    if frame < 60:
        stage = "① Roll 绕 X轴旋转"
    elif frame < 120:
        stage = "② Pitch 绕 Y轴旋转"
    else:
        stage = "③ Yaw 绕 Z轴旋转"
    ax.text2D(0.05, 0.78, stage, transform=ax.transAxes, color='k')

# ----------------------------------------
# 主程序
# ----------------------------------------
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

ani = FuncAnimation(fig, update, frames=180, interval=60, repeat=True)
plt.show()
