import cv2
import numpy as np
import pandas
from pandas import DataFrame

def search(image):

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
    grid_contours = [cnt.tolist() for cnt in grid_contours]

    return grid_contours

