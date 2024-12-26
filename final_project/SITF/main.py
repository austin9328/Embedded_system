import cv2
import numpy as np

# 讀取主圖像
image = cv2.imread('test_p.jpg', cv2.IMREAD_GRAYSCALE)

# 模板圖片路徑
template_paths = ['template_0.jpg', 'template_1.jpg', 'template_2.jpg', 'template_22.jpg', 
'template_3.jpg', 'template_32.jpg', 'template_4.jpg', 'template_42.jpg', 'template_5.jpg', 'template_52.jpg']

# 對應每個模板的顏色 (BGR格式)
colors = [
    (172, 10, 127),  # 深紫紅色
    (140, 47, 170),  # 紫色
    (196, 151, 117), # 褐色
    (22, 183, 192),  # 青綠色
    (204, 33, 216),  # 亮紫色
    (67, 179, 78),   # 鮮綠色
    (154, 251, 82),  # 黃綠色
    (162, 219, 195), # 淺綠色
    (118, 125, 212), # 淡藍色
    (53, 233, 38)    # 青綠色
]

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


    # 標記匹配區域
    color = colors[idx % len(colors)]  # 循環使用顏色
    for pt in zip(*loc[::-1]):
        cv2.rectangle(output_image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), color, 2)

# 儲存結果圖像
cv2.imwrite('Detected_Matches_Colored.jpg', output_image)