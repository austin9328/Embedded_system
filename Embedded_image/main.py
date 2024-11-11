import cv2
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
import sobel
import hsv
import search
import lbp
import histogram 
import one_norm_dist
import labeling

def main():
    radius = 1
    n_points = 8  
    patch_size = 16  
    # 讀取影像
    image = cv2.imread('test.jpg')
    image_gray = cv2.imread('test.jpg',0)
    height, width = image_gray.shape
    #################################
    #HSV
    hsv_img = hsv.hsv(image)
    cv2.imwrite('hsv.jpg',hsv_img)
    #################################
    #sobel
    Sobel_img = sobel.sobel_and_morphology(image_gray)
    cv2.imwrite('sobel.jpg',Sobel_img)

    #lbp
    lbp_img = lbp.calculate_lbp(Sobel_img, radius, n_points)
    cv2.imwrite('lbp.jpg',lbp_img)

    #search
    if image_gray is None:
        raise ValueError(f"Error: unable to load image '{lbp_img}'")
   
    # 將影像分割成小區塊
    patches = search.split_image_into_patches(image_gray, patch_size)

    #histogram
    mask = np.zeros_like(image_gray)  # 創建與影像相同尺寸的 mask
    #histogram找前三大(設定閥值)
    his = histogram.calculate_histogram(lbp_img)
    hist = histogram.plot_histogram(his)
    top3 = histogram.find_top_three(his)
    th = int(sum(top3)/3)
    
    rows, cols = image_gray.shape
    for i in range(0, rows, patch_size):
        for j in range(0, cols, patch_size):
            patch1 = image_gray[i:i+patch_size, j:j+patch_size]
            if i+patch_size < rows and j+patch_size < cols:
                # 與右邊的區塊進行比較
                patch2 = image_gray[i:i+patch_size, j+patch_size:j+2*patch_size]
                hist1 = histogram.calculate_histogram(patch1)
                hist2 = histogram.calculate_histogram(patch2)               
                if one_norm_dist.calculate_1_norm_distance(hist1, hist2,th) == 1:
                    mask[i:i+patch_size, j:j+patch_size] = 1  # 標記需要上色的區域
                
    #labeling
    colored_image = labeling.label_similar_areas(image, mask)
    
    # 顯示
    # 原始影像和著色後的影像
    cv2.imshow('Colored Image', colored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
