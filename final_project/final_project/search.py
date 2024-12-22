import cv2
import numpy as np
import pandas
from pandas import DataFrame

def search():
    image = cv2.imread('./photo/morphology.jpg', cv2.IMREAD_GRAYSCALE)
    # 二值化
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    contours,hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    test_image = np.zeros_like(image)
    cv2.drawContours(test_image, contours, -1, (255, 255, 0), 2)
    #cv2.imshow("123",test_image)
    #for i, cnt in enumerate(contours):
        #area = cv2.contourArea(cnt)
        #print(f"Contour {i}: Area = {area}")

    grid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 3100 < cv2.contourArea(cnt) < 7000:  # 篩選符合面積
            grid_contours.append(cnt)
    print(grid_contours)

    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, grid_contours, -1, (0, 255, 0), 2)
    cv2.imwrite('search.jpg',output)
    # 將 `ndarray` 轉換為列表 
    #grid_contours = [cnt.tolist() for cnt in grid_contours]

    return grid_contours


def sort_contours(grid_contours):
    # 將輪廓按y座標分組（考慮一定的容差）
    def get_row_index(y, tolerance=20):
        return int(y / tolerance)
        
    # 獲取所有矩形的邊界
    boxes = [cv2.boundingRect(c) for c in grid_contours]
    
    # 按y座標分組
    rows = {}
    for i, (x, y, w, h) in enumerate(boxes):
        row_idx = get_row_index(y)
        if row_idx not in rows:
            rows[row_idx] = []
        rows[row_idx].append((grid_contours[i], x))
    
    # 對每一行按x座標排序
    sorted_contours = []
    for row_idx in sorted(rows.keys()):
        row_contours = rows[row_idx]
        row_contours.sort(key=lambda x: x[1])  # 按x座標排序
        sorted_contours.extend([cnt for cnt, _ in row_contours])
    
    return sorted_contours



if __name__ == "__main__":
    labeled_image = search()
