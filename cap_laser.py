import cv2
import numpy as np
import time

# Mở webcam
cap = cv2.VideoCapture(0)

# Kiểm tra xem webcam có mở được không
if not cap.isOpened():
    print("Không thể mở webcam")
else:
    print("Đang nhận diện laser đỏ. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình")
            break

        # Chuyển đổi sang không gian màu HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Định nghĩa phạm vi màu đỏ trong HSV
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Tạo mặt nạ cho màu đỏ
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        # Tìm các vùng có màu đỏ
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Nếu phát hiện laser đỏ
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 50:  # Ngưỡng diện tích để tránh nhiễu
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    timestamp = int(time.time())
                    filename = f"/home/duypc/laser_detected_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Đã chụp ảnh: {filename}")
                    time.sleep(2)  # Tránh chụp quá nhiều ảnh liên tục

        # Hiển thị khung hình
        cv2.imshow("Laser Detection", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
