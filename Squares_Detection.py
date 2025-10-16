import cv2
import numpy as np

# Đọc ảnh
image = cv2.imread("target10.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Áp dụng ngưỡng để tạo ảnh nhị phân
_, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

# Tìm các contour
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Lọc các hình vuông
squares = []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    if len(approx) == 4 and area > 100 and cv2.isContourConvex(approx):
        # Kiểm tra tỷ lệ cạnh để đảm bảo là hình vuông
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.8 < aspect_ratio < 1.2:
            squares.append(approx)

# Vẽ các hình vuông lên ảnh
for square in squares:
    cv2.drawContours(image, [square], -1, (0, 255, 0), 2)

# Hiển thị số lượng hình vuông tìm được
print(f"Tìm thấy {len(squares)} hình vuông.")

# Hiển thị ảnh kết quả
cv2.imshow("Detected Squares", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
