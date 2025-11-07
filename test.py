import torch
import requests
import io
from PIL import Image
from super_gradients.training import models
from super_gradients.common.object_names import Models

def verify_environment():
    print("--- BẮT ĐẦU XÁC MINH MÔI TRƯỜNG ---")

    # 1. Xác minh PyTorch và CUDA
    print("\n")
    if torch.cuda.is_available():
        print(f"  Trạng thái: THÀNH CÔNG")
        print(f"  CUDA khả dụng: {torch.cuda.is_available()}")
        print(f"  Phiên bản CUDA của PyTorch: {torch.version.cuda}")
        print(f"  Tên GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print(f"  Trạng thái: THẤT BẠI. Không tìm thấy CUDA.")
        print("  Vui lòng kiểm tra lại Bước 2.3 (Trình điều khiển) và 3.3 (Cài đặt PyTorch).")
        return

    # 2. Tải mô hình YOLO-NAS
    print("\n")
    try:
        model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco").to(device)
        print("  Trạng thái: THÀNH CÔNG. Tải mô hình và chuyển sang GPU.")
    except Exception as e:
        print(f"  Trạng thái: THẤT BẠI. Lỗi khi tải mô hình: {e}")
        return

    # 3. Tải hình ảnh mẫu
    print("\n")
    image_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        image.save("temp_bus.jpg")
        print(f"  Trạng thái: THÀNH CÔNG. Đã tải và lưu 'temp_bus.jpg'.")
    except Exception as e:
        print(f"  Trạng thái: THẤT BẠI. Không thể tải hình ảnh: {e}")
        return

    # 4. Chạy suy luận (Inference)
    print("\n")
    confidence_threshold = 0.35
    try:
        predictions = model.predict("temp_bus.jpg", conf=confidence_threshold)
        print(f"  Trạng thái: THÀNH CÔNG. Suy luận hoàn tất.")

        # 5. Hiển thị kết quả
        print("\n")
        print("  Đang hiển thị hình ảnh với các hộp giới hạn (bounding boxes)...")
        predictions.show()
        print("  Đã lưu hình ảnh kết quả vào 'temp_bus.jpg_NAS_S_pred.jpg'")

        prediction_objects = list(predictions)
        print(f"\n  Các đối tượng được phát hiện: {len(prediction_objects.prediction.bboxes_xyxy)}")
        if len(prediction_objects.prediction.bboxes_xyxy) > 0:
            print("\n>>> XÁC MINH HOÀN TẤT: Toàn bộ hệ thống (Python 3.10, CUDA 12.1, PyTorch, super-gradients) đang hoạt động chính xác!")
        else:
            print("\n>>> XÁC MINH CÓ ĐIỀU KIỆN: Mã chạy nhưng không phát hiện đối tượng. Kiểm tra ngưỡng tin cậy.")

    except Exception as e:
        print(f"  Trạng thái: THẤT BẠI. Lỗi trong quá trình suy luận: {e}")

if __name__ == "__main__":
    verify_environment()