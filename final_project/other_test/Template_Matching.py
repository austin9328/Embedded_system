import cv2
import numpy as np

# 讀取主圖像
image = cv2.imread('./input/test_p.jpg', cv2.IMREAD_GRAYSCALE)

# 模板圖片路徑
template_paths = ['./template/template_11.jpg', './template/template_1.jpg', 
                  './template/template_2.jpg', './template/template_22.jpg',
                  './template/template_3.jpg', './template/template_32.jpg', './template/template_4.jpg', './template/template_42.jpg',
                  './template/template_5.jpg', './template/template_52.jpg']

# 對應每個模板的數字
template_numbers = ['1', '1', '2', '2', '3', '3', '4', '4', '5', '5']

# 讀取模板並存儲在列表中
template_images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in template_paths]

# 圖像二值化處理（主圖像）
_, image_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 原圖轉回彩色
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for idx, template in enumerate(template_images):
    # 對模板圖像進行二值化處理
    _, template_thresh = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
    # 使用模板匹配
    result = cv2.matchTemplate(image_thresh, template_thresh, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7  # 設定匹配的相似度閾值
    loc = np.where(result >= threshold)

    # 在匹配區域標記對應數字
    for pt in zip(*loc[::-1]):
        # 繪製數字，座標為匹配位置，顏色為黑色，字體大小為 1
        cv2.putText(output_image, template_numbers[idx], (pt[0], pt[1] + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# 儲存結果圖像
cv2.imwrite('./result/Matches_result.jpg', output_image)
