import cv2

img = cv2.imread("test_chessboard.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pattern_size = (11, 8)  # 改成你的真实内角点数量
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
print("检测结果:", ret, "角点数:", 0 if not ret else len(corners))
