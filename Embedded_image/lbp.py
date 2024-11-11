import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

def calculate_lbp(image, radius=1, n_points=8):
    """
    使用NumPy向量化操作優化的LBP計算
    
    參數:
    image: 輸入的灰度圖像
    radius: 圓形鄰域的半徑
    n_points: 圓形鄰域上採樣點的數量
    
    返回:
    lbp_image: LBP特徵圖像
    """
    rows = image.shape[0]
    cols = image.shape[1]
    
    # 生成圓形鄰域的坐標
    angles = 2 * np.pi * np.arange(n_points) / n_points
    x = radius * np.cos(angles)
    y = -radius * np.sin(angles)
    
    # 獲取鄰域坐標的四個參考點（用於雙線性插值）
    x1 = np.floor(x).astype(int)
    x2 = np.ceil(x).astype(int)
    y1 = np.floor(y).astype(int)
    y2 = np.ceil(y).astype(int)
    
    # 計算插值權重
    fx = x - x1
    fy = y - y1
    
    # 準備插值權重矩陣
    w1 = (1 - fx) * (1 - fy)
    w2 = fx * (1 - fy)
    w3 = (1 - fx) * fy
    w4 = fx * fy
    
    # 初始化輸出圖像
    lbp_image = np.zeros((rows, cols), dtype=np.uint8)
    
    # 填充圖像以處理邊界
    padded_image = np.pad(image, ((radius, radius), (radius, radius)), 'edge')
    
    # 在中心區域計算LBP
    for i in range(n_points):
        # 計算四個參考點的鄰域值
        n1 = padded_image[radius+x1[i]:rows+radius+x1[i], 
                         radius+y1[i]:cols+radius+y1[i]]
        n2 = padded_image[radius+x2[i]:rows+radius+x2[i], 
                         radius+y1[i]:cols+radius+y1[i]]
        n3 = padded_image[radius+x1[i]:rows+radius+x1[i], 
                         radius+y2[i]:cols+radius+y2[i]]
        n4 = padded_image[radius+x2[i]:rows+radius+x2[i], 
                         radius+y2[i]:cols+radius+y2[i]]
        
        # 雙線性插值
        neighbor = w1[i]*n1 + w2[i]*n2 + w3[i]*n3 + w4[i]*n4
        
        # 比較中心像素和鄰域像素
        center = padded_image[radius:radius+rows, radius:radius+cols]
        lbp_image += (neighbor >= center).astype(np.uint8) << i
    
    return lbp_image

# 使用示例
def main():
    # 讀取圖像
    image = cv2.imread('sobel.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("無法讀取圖像")
    '''
    # 比較原始版本和優化版本的執行時間
    import time
    
    start_time = time.time()
    opt_time = time.time() - start_time
    print(f"優化版本執行時間: {opt_time:.4f} 秒")
    '''
    lbp_result = calculate_lbp(image)
    # 顯示結果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(lbp_result, cmap='gray')
    plt.title('LBP Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return lbp_result

if __name__ == "__main__":
    main()

'''
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
def calculate_lbp(image, radius=1, n_points=8):
    """
    計算圖像的LBP特徵
    
    參數:
    image: 輸入的灰度圖像
    radius: 圓形鄰域的半徑
    n_points: 圓形鄰域上採樣點的數量
    
    返回:
    lbp_image: LBP特徵圖像
    """
    rows = image.shape[0]
    cols = image.shape[1]
    
    # 初始化輸出的LBP圖像
    lbp_image = np.zeros((rows, cols), dtype=np.uint8)
    
    # 為了處理邊界，我們跳過邊緣像素
    for i in range(radius, rows-radius):
        for j in range(radius, cols-radius):
            # 獲取中心像素值
            center = image[i, j]
            binary_code = 0
            
            # 計算每個鄰域點的值
            for p in range(n_points):
                # 計算採樣點的坐標
                r = radius
                x = i + r * np.cos(2 * np.pi * p / n_points)
                y = j - r * np.sin(2 * np.pi * p / n_points)
                
                # 取整
                x1 = int(np.floor(x))
                x2 = int(np.ceil(x))
                y1 = int(np.floor(y))
                y2 = int(np.ceil(y))
                
                # 雙線性插值
                fx = x - x1
                fy = y - y1
                w1 = (1 - fx) * (1 - fy)
                w2 = fx * (1 - fy)
                w3 = (1 - fx) * fy
                w4 = fx * fy
                
                # 計算採樣點的值
                neighbor = w1 * image[x1, y1] + w2 * image[x2, y1] + \
                          w3 * image[x1, y2] + w4 * image[x2, y2]
                
                # 比較中心像素和鄰域像素
                if neighbor >= center:
                    binary_code |= (1 << p)
            
            # 將二進制碼設置為該像素的LBP值
            lbp_image[i, j] = binary_code
            
    return lbp_image

# 使用示例：
def main():
    # 讀取圖像
    image = cv2.imread('sobel.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("無法讀取圖像")
    
    # 計算LBP
    lbp_result = calculate_lbp(image)
    
    # 顯示結果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(lbp_result, cmap='gray')
    plt.title('LBP Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 計算LBP直方圖
    hist = cv2.calcHist([lbp_result], [0], None, [256], [0, 256])
    
    # 繪製直方圖
    plt.figure(figsize=(10, 4))
    plt.plot(hist)
    plt.title('LBP Histogram')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    plt.show()
    
    return lbp_result, hist

if __name__ == "__main__":
    main()

'''