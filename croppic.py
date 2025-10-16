import cv2
import numpy as np

# Load the image
image = cv2.imread("real-target9.jpg")

def contours_gray (image_path):
    # Apply Gaussian blur and edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contours_red(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for bright red (laser dot)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Reduce noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_square_centers(image):
    centers = []
    square_contours = []

    for cnt in contours_gray(image):
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            square_contours.append(approx)

    for square in square_contours:
        coords = square.reshape(4, 2)
        center = np.mean(coords, axis=0)
        centers.append(center)
    # Format: top-left, top-right, bottom-right, bottom-left
    centers_array = np.array([centers[2], centers[3], centers[1], centers[0]], dtype="float32")
    return centers_array

pts = find_square_centers(image)
# Compute width and height of the new image
width = int(np.linalg.norm(pts[0] - pts[1]))
height = int(np.linalg.norm(pts[0] - pts[3]))
# Destination points for perspective transform
dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
# Compute the perspective transform matrix and apply it
M = cv2.getPerspectiveTransform(pts, dst)
cropped = cv2.warpPerspective(image, M, (width, height))
    
def square_center(image):
# Chuyển ảnh sang nhị phân (0 là nền, 255 là đối tượng)
    _, binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
# Tìm tọa độ của các điểm ảnh không phải nền
    coordinates = np.column_stack(np.where(binary > 0))
# Tính trung bình tọa độ để xác định tâm
    s_center = np.mean(coordinates, axis=0)
    return s_center
s_center = square_center(cropped)
print(f"Tâm của ảnh nằm tại tọa độ: ({int(s_center[0])}, {int(s_center[1])})")

# Draw detected laser dot
for cnt in contours_red(cropped):
    area = cv2.contourArea(cnt)
    if area > 50:  # Filter small noise
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        print(f"Laser dot detected at position: x={center[0]}, y={center[1]}")
        #position = f"Laser Dot ({center[0]}, {center[1]})"
        # Image center
        #img_center = (square_center[0] // 2, square_center[1] // 2)

        # Distance from center
        dx = center[0] - s_center[0]
        dy = center[1] - s_center[1]
        distance = (dx**2 + dy**2)**0.5
        # Define scoring zones (you can customize these)
        score = 0
        if distance < 50:
            score=10
        elif distance < 110:
            score=9
        elif distance < 170:
            score=8
        elif distance < 230:
            score=7
        elif distance < 300:
            score=6
        else:
            score=1
        score = f"Score: ({score})"
        print(f"distance: {distance} ; Score: ({score})")
        #cv2.circle(image, center, int(radius), (0, 255, 0), 2)
        cv2.putText(cropped, score, (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Show result
cv2.imshow("Center of the large square", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
