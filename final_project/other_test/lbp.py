import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from collections import defaultdict

def extract_lbp_features(image, radius=1, n_points=8, n_bins=10):
    """
    提取 LBP 直方圖特徵
    """
    lbp = local_binary_pattern(image, P=n_points, R=radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist

def preprocess_image(image_path, size=(64, 64)):
    """
    圖片預處理：灰度化 + 大小標準化 + 二值化
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"無法讀取圖片：{image_path}")
    image = cv2.resize(image, size)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:  # 確保前景為白色
        binary = 255 - binary
    return binary

def train_dynamic_thresholds(sample_paths, labels, radius=1, n_points=8, n_bins=10):
    """
    根據樣本動態生成筆劃數閾值範圍
    """
    features_by_stroke = defaultdict(list)

    for path, label in zip(sample_paths, labels):
        image = preprocess_image(path)
        hist = extract_lbp_features(image, radius, n_points, n_bins)
        features_by_stroke[label].append(hist)
    
    thresholds = {}
    for strokes, histograms in features_by_stroke.items():
        stacked_hist = np.vstack(histograms)
        mean_hist = np.mean(stacked_hist, axis=0)
        std_hist = np.std(stacked_hist, axis=0)
        thresholds[strokes] = (mean_hist, std_hist)
    
    return thresholds

def predict_stroke_count(image_path, thresholds, radius=1, n_points=8, n_bins=10):
    """
    使用動態閾值預測筆劃數
    """
    image = preprocess_image(image_path)
    test_hist = extract_lbp_features(image, radius, n_points, n_bins)
    
    best_match = None
    min_distance = float("inf")

    for strokes, (mean_hist, std_hist) in thresholds.items():
        # 計算測試直方圖與樣本的歐幾里得距離
        distance = np.linalg.norm((test_hist - mean_hist) / (std_hist + 1e-6))
        if distance < min_distance:
            min_distance = distance
            best_match = strokes
    
    return best_match


# 測試代碼
if __name__ == "__main__":
    # 樣本圖片路徑和對應的筆劃數
    sample_paths = [
        './template/template_0.jpg',
        './template/template_11.jpg',
        './template/template_2.jpg',
        './template/template_3.jpg',
        './template/template_4.jpg',
        './template/template_5.jpg',
    ]
    sample_labels = [0,1, 2, 3, 4, 5]

    # 訓練動態閾值
    n_bins = 10
    thresholds = train_dynamic_thresholds(sample_paths, sample_labels, n_bins=n_bins)

    # 測試圖片
    for i in range(0,58):
        test_image_path = f'./cropped/cropped_image_{i}.jpg'
        predicted_strokes = predict_stroke_count(test_image_path, thresholds, n_bins=n_bins)
        print(f"{i}:預測的筆劃數：{predicted_strokes}")
