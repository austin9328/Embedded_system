import cv2
import numpy as np
import time
start_time = time.time()
image1 = cv2.imread("./input/non_marked.jpg") 
image2 = cv2.imread("./input/marked2.jpg")      


if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Farneback
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None, 
    pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# 計算光流大小和方向
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

# 針對劃保留小幅度變化
threshold_min = 1.0  # 最小光流幅度
threshold_max = 29.0  # 最大光流幅度
motion_mask = cv2.inRange(magnitude, threshold_min, threshold_max)
# 擴大光流的影響範圍，保證細線條能被捕捉
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)


diff = cv2.absdiff(gray1, gray2)# 疊加差值，進一步過濾
_, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)


final_mask = cv2.bitwise_and(motion_mask, diff_mask)#結合光流與差值

# find劃記區域
contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = image2.copy()
min_area = 30  # 最小面積閾值

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:  # 過濾小面積
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 綠框
        print(f"發現劃記區域：x={x}, y={y}, w={w}, h={h}, 面積={area}")
end_time = time.time()

execution_time = end_time - start_time
print(f"程式執行時間：{execution_time:.4f} 秒")
cv2.imshow("Motion Mask", motion_mask)
cv2.imshow("Difference Mask", diff_mask)
cv2.imshow("Final Mask", final_mask)
cv2.imshow("Detected Changes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
