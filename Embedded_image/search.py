'''
import cv2
import numpy as np
from collections import deque
from matplotlib import pyplot as plt

def bfs_image_segmentation(image, start, patch_size, threshold=10):
    """
    使用 BFS (廣度優先搜索) 方法對影像進行分割，提取指定區域的小塊。
    
    參數:
    image (numpy.ndarray): 輸入的灰度影像
    start (tuple): 起始像素座標 (x, y)
    patch_size (int): 要提取的小塊大小
    threshold (int): 判斷相鄰像素是否屬於同一區域的灰度值閾值
    
    返回:
    numpy.ndarray: 提取的小塊影像
    """
    rows, cols = image.shape  # 獲取影像的行和列數
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError("起始像素超出影像範圍")  # 檢查起始像素是否在影像範圍內

    mask = np.zeros((rows, cols), dtype=np.uint8)  # 創建一個與影像同尺寸的掩膜，用於標記已處理的區域
    queue = deque([start])  # 初始化佇列，將起始像素加入佇列
    start_value = int(image[start])  # 獲取起始像素的灰度值
    visited = np.zeros((rows, cols), dtype=bool)  # 創建一個布爾矩陣，用於記錄是否訪問過某像素
    visited[start] = True  # 標記起始像素已被訪問
    patch_list = []  # 初始化一個列表，用於存儲小塊內的像素座標

    while queue:  # 當佇列不為空時
        x, y = queue.popleft()  # 取出佇列中的第一個元素（當前像素座標）
        patch_list.append((x, y))  # 將當前像素座標加入小塊列表
        mask[x, y] = 255  # 在掩膜中標記當前像素
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy  # 計算鄰近像素的座標
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                if abs(int(image[nx, ny]) - start_value) < threshold:  # 判斷鄰近像素是否屬於同一區域
                    queue.append((nx, ny))  # 如果是，將鄰近像素加入佇列
                    visited[nx, ny] = True  # 標記鄰近像素已被訪問
        
        # 如果已達到區域大小，則停止
        if len(patch_list) >= patch_size * patch_size:
            break

    # 提取小區塊內的像素值
    patch_image = np.zeros((patch_size, patch_size), dtype=image.dtype)  # 創建一個與小塊大小相同的空矩陣
    for idx, (x, y) in enumerate(patch_list):
        if idx < patch_size * patch_size:
            patch_image[idx // patch_size, idx % patch_size] = image[x, y]  # 將小塊內的像素值填充到小塊影像中

    return patch_image  # 返回提取的小塊影像

def main():
    # 讀取影像
    image_file = 'lbp.jpg'
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: unable to load image '{image_file}'")
    
    # 設定分割區塊大小
    patch_size = 16  # 例如，16x16 的區域

    # 將影像分割成多個小區塊並存儲在列表中
    patches = []
    rows, cols = image.shape
    for i in range(0, rows, patch_size):
        for j in range(0, cols, patch_size):
            patch = bfs_image_segmentation(image, (i, j), patch_size)
            patches.append(patch)
    
    # 顯示原始影像和部分小區塊
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    for idx, patch in enumerate(patches[:10]):  # 只顯示前10個小區塊
        plt.subplot(3, 10, idx + 11)
        plt.imshow(patch, cmap='gray')
        plt.title(f'Patch {idx + 1}')
        plt.axis('off')

    plt.suptitle('Original Image and Patches')
    plt.show()

if __name__ == '__main__':
    main()

'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

def split_image_into_patches(image, patch_size):
    """
    將影像分割成多個小區塊並存儲在列表中。
    
    參數:
    image (numpy.ndarray): 輸入的灰度影像
    patch_size (int): 每個小區塊的大小
    
    返回:
    list: 包含所有小區塊的列表
    """
    patches = []
    rows, cols = image.shape
    
    for i in range(0, rows, patch_size):
        for j in range(0, cols, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    
    return patches

def main():
    # 讀取影像
    image_file = 'lbp.jpg'
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: unable to load image '{image_file}'")
    
    # 設置分割區塊大小
    patch_size = 32  # 例如，16x16 的區域

    # 將影像分割成小區塊
    patches = split_image_into_patches(image, patch_size)

    # 顯示原始影像和部分小區塊
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    for idx, patch in enumerate(patches[:10]):  # 只顯示前10個小區塊
        plt.subplot(3, 10, idx + 11)
        plt.imshow(patch, cmap='gray')
        plt.title(f'Patch {idx + 1}')
        plt.axis('off')

    plt.suptitle('Original Image and Patches')
    plt.show()

if __name__ == '__main__':
    main()
