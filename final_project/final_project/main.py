import cv2
import numpy as np
import gray_img
import threshold
import morphology
import find_line
import sobel


# 讀取輸入圖像
img = cv2.imread('./photo/test_2.jpg')  # 讀取名稱為'test.jpg'的圖像文件

# gray_img
gray = gray_img.bgr_to_gray(img)

# threshold
binary = threshold.threshold(gray, 127, 255)

# morphology
kernel = np.ones((5, 5), np.uint8)  # 創建一個 5x5 的核
opened_custom = morphology.morphological_opening(binary // 255, kernel)

# 進行二值化處理
binary = threshold.threshold(opened_custom, 127, 255)

# 找到輪廓
contours, hierarchy = cv2.findContours(opened_custom, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 發現圖像中的輪廓

# 在空白圖像上繪製輪廓
drawing = np.zeros_like(img)  # 創建一個與原始圖像大小相同的空白圖像
cv2.drawContours(drawing, contours, -1, (0, 255, 0), 2)  # 用綠色線條繪製所有的輪廓

# 保存輸出圖像
cv2.imwrite('./photo/output_image.jpg', drawing)  # 將繪製了輪廓的圖像保存為'output_image.jpg'

#search 找出格子

photo_list = '00000'
price_list = 'dgddsvsd'#每格的價格
#統計數量
all_line = []
for i in range(1,10):
    sobelx ,sobely = sobel.sobel_and_morphology(photo_list[i])
    # 創建一個 5x5 的結構元素
    kernel = np.ones((11, 11), np.uint8)
    # 進行二值膨脹操作
    sobelx = cv2.dilate(sobelx,kernel,iterations=1)
    sobely = cv2.dilate(sobely,kernel,iterations=1)
    x_line = find_line.count_lines(sobelx,offset=10)
    y_line = find_line.count_lines(sobely,offset=10)
    all_line[i] = x_line + y_line
#相乘 計算每項價錢
result = [a * b for a, b in zip(all_line, price_list)]
#每項相加
total = sum(result)


