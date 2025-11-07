import cv2
import os

def extract_frame(video_path, frame_number=108):
    """
    Lấy frame thứ `frame_number` từ video và lưu ra file ảnh.
    Trả về: đường dẫn ảnh (hoặc None nếu lỗi)
    """
    if not os.path.exists(video_path):
        print(f"❌ File không tồn tại: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_number >= total_frames:
        print(f"⚠️ Frame {frame_number} vượt quá tổng số frame ({total_frames})")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        print(f"❌ Không thể đọc frame {frame_number}")
        return None

    # Tạo đường dẫn lưu ảnh
    output_path = os.path.join(
        os.path.dirname(video_path),
        f"frame_{frame_number}.jpg"
    )

    cv2.imwrite(output_path, frame)
    print(f"✅ Frame {frame_number} đã được lưu tại: {output_path}")

    return output_path


frame_path = extract_frame(
    r"D:\zalo_ai\observing\train\samples\Lifering_0\drone_video.mp4",
    frame_number=108
)
print("Đường dẫn ảnh:", frame_path)