import cv2, os

root = 'D:\zalo_ai\output\yolo_dataset\images'  # chỉnh lại đường dẫn
for fname in os.listdir(root):
    path = os.path.join(root, fname)
    img = cv2.imread(path)
    if img is None:
        print("❌ Ảnh lỗi hoặc không đọc được:", path)
