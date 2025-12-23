
import cv2
import numpy as np

# 加载新标定的相机内参
fs = cv2.FileStorage('camera_intrinsics.yaml', cv2.FILE_STORAGE_READ)
K_original = fs.getNode('K').mat()
dist_original = fs.getNode('distCoeffs').mat().flatten()
fs.release()

print(f"标定的原始内参: fx={K_original[0,0]:.1f}, fy={K_original[1,1]:.1f}")

# 根据测量结果计算修正系数
# 原始测量 647mm，实际 600mm，修正系数 = 600/647 ≈ 0.927
correction_factor = 600 / 647

# 修正后的内参矩阵
K_corrected = K_original.copy()
K_corrected[0, 0] *= correction_factor  # fx
K_corrected[1, 1] *= correction_factor  # fy

print(f"修正后的内参: fx={K_corrected[0,0]:.1f}, fy={K_corrected[1,1]:.1f}")

board_size = (11, 8)
square_size = 0.02073

# 打开相机测试
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n距离测量测试")
print("显示三种测量结果:")
print("  红色: 原始标定内参")
print("  绿色: 修正后内参 (factor=600/647)")
print("  蓝色: 手动调整焦距 (按+/-调整)")
print("\n按键: +/- 调整焦距, Q 退出")

# 手动调整的焦距
manual_fx = K_original[0, 0]

objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
objp *= square_size

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size, None)
    
    display = frame.copy()
    
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display, board_size, corners, found)
        
        imgp = corners.reshape(-1, 2)
        
        # 原始内参
        _, rvec1, tvec1 = cv2.solvePnP(objp, imgp, K_original, dist_original)
        dist1 = np.linalg.norm(tvec1) * 1000
        
        # 修正内参
        _, rvec2, tvec2 = cv2.solvePnP(objp, imgp, K_corrected, dist_original)
        dist2 = np.linalg.norm(tvec2) * 1000
        
        # 手动调整的内参
        K_manual = K_original.copy()
        K_manual[0, 0] = manual_fx
        K_manual[1, 1] = manual_fx * (K_original[1,1] / K_original[0,0])  # 保持纵横比
        _, rvec3, tvec3 = cv2.solvePnP(objp, imgp, K_manual, dist_original)
        dist3 = np.linalg.norm(tvec3) * 1000
        
        cv2.putText(display, f"Original (fx={K_original[0,0]:.0f}): {dist1:.0f}mm", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, f"Corrected (fx={K_corrected[0,0]:.0f}): {dist2:.0f}mm", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Manual (fx={manual_fx:.0f}): {dist3:.0f}mm", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 显示Z轴距离 (通常更准确)
        cv2.putText(display, f"Z-dist: Orig={tvec1[2][0]*1000:.0f} Corr={tvec2[2][0]*1000:.0f} Man={tvec3[2][0]*1000:.0f}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(display, "No chessboard", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.putText(display, "Press +/- to adjust fx, Q to quit", (10, display.shape[0]-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Distance Test', display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        manual_fx += 10
        print(f"Manual fx: {manual_fx:.0f}")
    elif key == ord('-') or key == ord('_'):
        manual_fx -= 10
        print(f"Manual fx: {manual_fx:.0f}")

cap.release()
cv2.destroyAllWindows()

print(f"\n最终手动调整的 fx = {manual_fx:.1f}")
print(f"如果这个值测量准确，可以用它更新相机内参文件")
