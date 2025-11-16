import cv2
import numpy as np
import glob
import os

# -------------------------------
# 参数配置
# -------------------------------
# 棋盘格行列数 (内角点数)
board_size = (11, 8)  # 横11格竖8格 => 88个角点
# 每个小方格的边长 (单位: 米)
square_size = 0.022  # 20mm

# 棋盘格图片所在文件夹
image_folder = "./calib_images"
# 输出文件
intrinsic_file = "camera_intrinsics.yaml"
extrinsic_file = "extrinsics.npy"

# -------------------------------
# 构造棋盘格世界坐标系
# -------------------------------
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
objp *= square_size

# 用于存储所有图片的3D点和2D点
objpoints = []  # 世界坐标（标定板坐标系）
imgpoints = []  # 图像坐标

# -------------------------------
# 读取标定图片
# -------------------------------
images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) +
                glob.glob(os.path.join(image_folder, "*.png")))

print(f"🧩 共找到 {len(images)} 张标定图片")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    if ret:
        objpoints.append(objp)
        # 亚像素级优化
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # 绘制角点显示
        cv2.drawChessboardCorners(img, board_size, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(100)
    else:
        print(f"⚠️ 未检测到角点：{fname}")

cv2.destroyAllWindows()

# -------------------------------
# 相机内参标定
# -------------------------------
print("📷 开始计算相机内参...")
ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("✅ 标定完成！")
print("内参矩阵 K：\n", K)
print("畸变系数：", distCoeffs.ravel())

# -------------------------------
# 保存内参
# -------------------------------
fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_WRITE)
fs.write("K", K)
fs.write("distCoeffs", distCoeffs)
fs.release()
print(f"💾 内参已保存到 {intrinsic_file}")

# -------------------------------
# 计算并保存每张图片的外参（T_target^cam）
# -------------------------------
extrinsics = []

for rvec, tvec in zip(rvecs, tvecs):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.squeeze()
    extrinsics.append(T)

extrinsics = np.array(extrinsics)
np.save(extrinsic_file, extrinsics)
print(f"💾 外参矩阵（每张图的 T_target^cam）已保存到 {extrinsic_file}")
