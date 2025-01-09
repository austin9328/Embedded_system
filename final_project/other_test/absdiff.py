import cv2
import numpy as np
import time
image1 = cv2.imread("./input/non_marked.jpg")  
image2 = cv2.imread("./input/marked2.jpg")      

start_time = time.time()

if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0])) # 將兩張圖片resize大小相同
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)
blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)

# 計算差值
diff = cv2.absdiff(blurred1, blurred2)

# 二值化
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

#除噪，減誤判
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = image2.copy()
min_area = 50  # 最小面積閾值

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:  # 過濾小面積誤判
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 綠色框
        print(f"發現差異區域：x={x}, y={y}, w={w}, h={h}, 面積={area}")
end_time = time.time()

execution_time = end_time - start_time
print(f"程式執行時間：{execution_time:.4f} 秒")
cv2.imshow("Difference", diff)
cv2.imshow("Thresholded", thresh)
cv2.imshow("Cleaned", cleaned)
cv2.imshow("Detected Changes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
