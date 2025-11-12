import numpy as np
from ik import ET, Robot

# ======== 你的原函数 ========
def create_so101_5dof():
    """
    创建 SO-101 5自由度机械臂模型
    """
    E1 = ET.Rz()      # shoulder_pan
    E2 = ET.tx(0.0612)
    E3 = ET.tz(0.0598)
    E4 = ET.tx(0.02943)
    E5 = ET.tz(0.05504)
    E6 = ET.Ry()      # shoulder_lift
    E7 = ET.tz(0.1127)
    E8 = ET.tx(0.02798)
    E9 = ET.Ry()      # elbow_flex
    E10 = ET.tx(0.13504)
    E11 = ET.tz(0.00519)
    E12 = ET.Ry()     # wrist_flex
    E13 = ET.tx(0.0593)
    E14 = ET.tz(0.00996)
    E15 = ET.Rx()     # wrist_roll

    ets = E1 * E2 * E3 * E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15

    qlim = np.array([
        [-1.91986, -1.74533, -1.69, -1.65806, -2.74385],
        [ 1.91986,  1.74533,  1.69,  1.65806,  2.84121]
    ])
    return Robot(ets, qlim)


# ======== 检查函数 ========
def check_handness(robot):
    """检查机械臂坐标系是否为右手系"""
    T = robot.fkine([0, 0, 0, 0, 0]) # 齐次矩阵
    R = T[:3, :3]
    x, y, z = R[:, 0], R[:, 1], R[:, 2]

    cross_xy = np.cross(x, y)
    dot_val = np.dot(cross_xy, z)

    print("末端齐次矩阵:\n", T)
    print("\nX 轴:", x)
    print("Y 轴:", y)
    print("Z 轴:", z)
    print("X×Y =", cross_xy)
    print("X×Y · Z =", dot_val)

    if dot_val > 0:
        print("\n✅ 当前为【右手系】")
    else:
        print("\n⚠️ 当前为【左手系】")


# ======== 主程序入口 ========
if __name__ == "__main__":
    robot = create_so101_5dof()
    check_handness(robot)
