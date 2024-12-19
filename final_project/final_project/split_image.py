import cv2

def resize_image(image_path, scale_factor, interpolation_method):
    """
    將圖像按比例放大或縮小。
    
    參數:
    - image_path: 原始圖像的路徑
    - scale_factor: 放大倍數（例如：4 表示放大 4 倍）
    - interpolation_method: 插值方法（cv2.INTER_NEAREST, cv2.INTER_LINEAR, 等）
    """
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: 無法讀取圖像，請檢查路徑。")
        return
    
    # 獲取原始圖像的尺寸
    original_height, original_width = image.shape[:2]
    print(f"原始圖像尺寸: {original_width}x{original_height}")
    
    # 計算新尺寸
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # 進行圖像插值
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation_method)
    print(f"放大後圖像尺寸: {new_width}x{new_height}")
    
    # 保存結果
    output_path = "./photo/resized_image.png"
    cv2.imwrite(output_path, resized_image)
    print(f"放大後的圖像已保存至: {output_path}")
    
    # 顯示圖像
    cv2.imshow("Original Image", image)
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用範例
image_path = "./photo/sobel.jpg"  # 替換為你的圖像路徑
scale_factor = 4          # 放大倍數，例如 4 表示 100x100 -> 400x400
interpolation_method = cv2.INTER_CUBIC  # 插值方法，可選 INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, 等

resize_image(image_path, scale_factor, interpolation_method)
