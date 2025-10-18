
import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Define red color range for laser dot detection in HSV
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for red color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contour is large enough to be a laser dot
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5:  # small threshold for laser dot
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if not captured:
                cv2.imwrite("/home/duypc/laser_detected.jpg", frame)
                captured = True
                print("Laser dot detected and image captured.")

    # Break loop after capture
    if captured:
        break

# Release camera
cap.release()
cv2.destroyAllWindows()

