import cv2
import os
import time

# -------------------------------
# 参数配置
# -------------------------------
save_dir = "./calib_images"     # 图片保存目录
file_prefix = "img_"            # 文件名前缀
cam_id = 0                      # 摄像头ID（USB相机一般为0，多个相机可改1,2...）
img_format = ".jpg"             # 图片格式
img_width, img_height = 1280, 720   # 图像分辨率
max_images = 20                 # 拍照张数上限（可修改）
preview_scale = 0.7             # 预览缩放比例

# -------------------------------
# 初始化相机
# -------------------------------
cap = cv2.VideoCapture(cam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

if not cap.isOpened():
    raise IOError("❌ 无法打开相机，请检查连接或相机ID。")

# 创建保存文件夹
os.makedirs(save_dir, exist_ok=True)

print("✅ 相机已打开，按 [空格] 拍照，按 [q] 退出。")
print(f"📁 图片将保存到：{os.path.abspath(save_dir)}")

# -------------------------------
# 主循环
# -------------------------------
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 图像读取失败！")
        break

    # 缩放显示
    display = cv2.resize(frame, (int(img_width*preview_scale), int(img_height*preview_scale)))

    # 显示拍摄计数
    cv2.putText(display, f"Captured: {count}/{max_images}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Camera Preview", display)

    # 键盘控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # 空格拍照
        filename = os.path.join(save_dir, f"{file_prefix}{count:02d}{img_format}")
        cv2.imwrite(filename, frame)
        print(f"📸 已保存: {filename}")
        count += 1
        time.sleep(0.3)  # 防抖
        if count >= max_images:
            print("✅ 已达到拍照上限。")
            break
    elif key == ord('q'):  # 退出
        print("🛑 手动退出。")
        break

cap.release()
cv2.destroyAllWindows()
