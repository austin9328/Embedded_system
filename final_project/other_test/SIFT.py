import cv2

def load_samples(sample_paths):
    sift = cv2.SIFT_create()
    sample_features = []

    for path in sample_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"無法讀取樣本影像：{path}")
        
        keypoints, descriptors = sift.detectAndCompute(image, None)
        sample_features.append(descriptors)

    return sample_features

def check_for_sample_match(image_path, sample_paths, sample_features, ratio_test_threshold=0.85, match_threshold=5):
    # 讀取點餐卡影像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("無法讀取影像。請檢查影像路徑是否正確。")

    # 初始化SIFT
    sift = cv2.SIFT_create()

    # 檢測特徵點和計算描述子
    keypoints, descriptors = sift.detectAndCompute(image, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    best_match_index = -1
    max_good_matches = 0

    for i, sample_descriptors in enumerate(sample_features):
        matches = bf.knnMatch(descriptors, sample_descriptors, k=2)
        
        # 應用Lowe's Ratio Test過濾特徵匹配
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_test_threshold * n.distance:
                good_matches.append(m)
        
        # 記錄最大良好匹配數量
        if len(good_matches) > max_good_matches:
            max_good_matches = len(good_matches)
            best_match_index = i

    # 判斷是否有足夠的良好匹配
    if max_good_matches >= match_threshold:
        return sample_numbers[best_match_index]
    
    else:
        return "都不是"

# 使用範例
sample_paths = [
    './template/template_11.jpg',
    './template/template_2.jpg',
    './template/template_3.jpg',
    './template/template_4.jpg',
    './template/template_5.jpg'
]
sample_numbers = ['1','2','3','4','5']

# 加載樣本特徵
sample_features = load_samples(sample_paths)

# 測試影像
for i in range(0,58):

    test_image_path = f'./cropped_non_line/cropped_{i}.jpg'
    result = check_for_sample_match(test_image_path, sample_paths, sample_features, ratio_test_threshold=0.85, match_threshold=5)
    print(f"檢測結果{i}：{result}")
