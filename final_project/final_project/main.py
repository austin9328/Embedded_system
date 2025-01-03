import cv2
import numpy as np
import gray_img
import threshold
import morphology
import find_line
import sobel
import search
import Flatten

# 讀取輸入圖像
img = cv2.imread('./photo/test_p.jpg')  # 讀取名稱為'test.jpg'的圖像文件

# 檢查圖像方向
if img.shape[1] > img.shape[0]:  # 如果圖像寬度大於高度，則旋轉圖像
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

#photo = Flatten.flatten(img)
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

#search 找出格子座標
xy_list = search.search()
xy_list = search.sort_contours(xy_list)
price_list = [0,280,280,280,280,280,280,280,280,280,
              350,350,350,160,160,160,160,160,160,
              160,160,160,180,180,180,200,200,200,200,
              200,200,200,200,200,220,220,220,320,320,
              320,320,320,320,450,450,450,80,80,80,
              0,0,0,0,30,50,20,0,0
              ]

print(xy_list)
''' 
# 創建原圖的副本，避免直接修改原圖
marked_image = img.copy()
for i, cnt in enumerate(xy_list):
    # 獲取邊界矩形的座標
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 繪製綠色矩形框，線寬為 2
    cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 添加序號標籤
    # 設置標籤文字
    label = f'#{i}'
    
    # 設置文字參數
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # 獲取文字大小
    (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # 計算文字的位置（左上角位置）
    text_x = x
    text_y = y - 5  # 將文字放在矩形框上方 5 個像素
    
    # 如果文字位置會超出圖像上邊界，則將文字放在矩形框內的上方
    if text_y < label_height:
        text_y = y + label_height + 5
    
    # 添加白色背景的文字標籤
    cv2.putText(marked_image, label, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

# 保存標記後的圖像
cv2.imwrite('./photo/marked_image_1.jpg', marked_image)
'''
# 創建一個空列表來存儲裁剪後的圖像
cropped_images = []

# 遍歷座標列表並裁剪圖像
for i, cnt in enumerate(xy_list):

    # 確保 cnt 是 NumPy 數組
    #cnt = np.array(cnt, dtype=np.int32)

    x,y,w,h = cv2.boundingRect(cnt)
    # 裁剪區域
    cropped_image = binary[y:y+h, x:x+w]
    cropped_images.append(cropped_image)
    
    # 可選：保存每個裁剪的圖像
    cv2.imwrite(f'./cropped/cropped_image_{i}.jpg', cropped_image)

print(type(cropped_image))

''' '''
#統計數量
all_line = []
for i in range(0,57):
    sobelx ,sobely = sobel.sobel_and_morphology(cropped_images[i])
    # 創建一個 5x5 的結構元素
    kernel = np.ones((3, 3), np.uint8)
    sobelx =threshold.threshold(sobelx, 127, 255)
    sobely =threshold.threshold(sobely,127,255)
    # 進行二值膨脹操作
    sobelx = cv2.dilate(sobelx,kernel,iterations=1)
    sobely = cv2.dilate(sobely,kernel,iterations=1)
    cv2.imwrite('a.jpg', sobelx)
    cv2.imwrite('b.jpg', sobely)
    x_line = find_line.count_lines(sobelx,offset=10)
    y_line = find_line.count_lines(sobely,offset=10)
    total = x_line + y_line -4 
    all_line.append(total)

#相乘 計算每項價錢
result = [a * b for a, b in zip(all_line, price_list)]
#每項相加
total = sum(result)
print(total)

