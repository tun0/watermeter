import cv2
import numpy as np
import sys

img = cv2.imread('/tmp/meter_ref.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Boost contrast with CLAHE before detection
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
blurred = cv2.GaussianBlur(enhanced, (7, 7), 2)

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=70,
    param1=40,
    param2=20,
    minRadius=45,
    maxRadius=100,
)

output = img.copy()
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    # Sort top-to-bottom then left-to-right (meter is rotated 90°)
    circles = sorted(circles, key=lambda c: (c[1], c[0]))
    print(f"Detected {len(circles)} circles:")
    for i, (x, y, r) in enumerate(circles):
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 4, (0, 100, 255), -1)
        cv2.putText(output, str(i), (x - 10, y - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        print(f"  [{i}] center=({x:4d}, {y:4d})  radius={r}")
else:
    print("No circles detected")

cv2.imwrite('/tmp/meter_detected.jpg', output)
print("Saved /tmp/meter_detected.jpg")
