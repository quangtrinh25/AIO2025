import os
import json
import cv2
import torch
from tqdm import tqdm
from super_gradients.training import models


def generate_predictions(model_path: str,
                         samples_dir: str,
                         output_file: str,
                         conf_threshold: float = 0.05,
                         save_debug_video: bool = True,
                         debug_video_dir: str = "output_videos"):
    """
    Chạy inference YOLO-NAS + tracking cho từng thư mục object trong samples_dir.
    Mỗi thư mục có:
        - object_images/ (3 ảnh mẫu)
        - drone_video.mp4
    """

    print(f"\n{'='*70}")
    print("Generating YOLO-NAS Tracking Predictions (with DEBUG info)")
    print(f"{'='*70}\n")

    # ===============================================================
    # 1️⃣ LOAD MODEL + CHECKPOINT
    # ===============================================================
    best_model = os.path.join(os.path.dirname(model_path), "ckpt_best.pth")
    if os.path.exists(best_model):
        model_path = best_model

    print(f"🔍 Loading model from: {model_path}")
    try:
        model = models.get("yolo_nas_s", num_classes=1, checkpoint_path=model_path)
        print("✅ Checkpoint loaded successfully!")
        print("🔎 Model head output shape:", model._modules.get('head'))

    except Exception as e:
        print(f"❌ Could not load checkpoint: {e}")
        print("⚠ Using base YOLO-NAS_S model without trained weights.")
        model = models.get("yolo_nas_s", num_classes=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"🚀 Model running on: {device}\n")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(debug_video_dir, exist_ok=True)

    # ===============================================================
    # 2️⃣ DUYỆT CÁC OBJECT
    # ===============================================================
    object_folders = [
        d for d in os.listdir(samples_dir)
        if os.path.isdir(os.path.join(samples_dir, d))
    ]
    print(f"📂 Found {len(object_folders)} objects to process.\n")

    predictions = []

    for obj_id in tqdm(object_folders, desc="Processing objects", unit="object"):
        obj_dir = os.path.join(samples_dir, obj_id)

        # tìm video thực sự (tự dò tên để tránh lỗi)
        video_path = None
        for f in os.listdir(obj_dir):
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(obj_dir, f)
                break

        if not video_path or not os.path.exists(video_path):
            print(f"⚠ No video found in {obj_id}, skipping.")
            predictions.append({"video_id": obj_id, "detections": []})
            continue

        print(f"\n🎥 Processing video: {obj_id}")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"   • Frames: {total_frames} | FPS: {fps:.2f}")
        cap.release()

        detections = []
        frame_num = 0
        total_bboxes = 0

        try:
            pred_gen = model.predict(
                video_path,
                conf=conf_threshold,
                tracker="bytetrack.yaml"
            )

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            out_path = os.path.join(debug_video_dir, f"{obj_id}_tracked.mp4")

            out_writer = None
            if save_debug_video:
                out_writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
                )

            # ===============================================================
            # 3️⃣ DUYỆT FRAME VÀ GHI LẠI DETECTIONS
            # ===============================================================
            frame_count = 0
            for frame_pred in pred_gen:
                ret, frame = cap.read()
                if not ret:
                    break

                pred = frame_pred.prediction
                bboxes = []

                if hasattr(pred, "bboxes_xyxy") and pred.bboxes_xyxy is not None:
                    for i in range(len(pred.bboxes_xyxy)):
                        x1, y1, x2, y2 = map(float, pred.bboxes_xyxy[i])
                        conf = float(pred.confidence[i]) if hasattr(pred, "confidence") else 1.0
                        if conf >= conf_threshold:
                            bboxes.append({
                                "frame": frame_num,
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2
                            })

                            if save_debug_video:
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(frame, f"{conf:.2f}",
                                            (int(x1), max(0, int(y1) - 5)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                if bboxes:
                    detections.append({"bboxes": bboxes})
                    total_bboxes += len(bboxes)

                if save_debug_video and out_writer is not None:
                    out_writer.write(frame)

                frame_num += 1
                frame_count += 1

            if save_debug_video and out_writer is not None:
                out_writer.release()
            cap.release()

            print(f"   • Processed {frame_count} frames, total detections: {total_bboxes}")

            predictions.append({
                "video_id": obj_id,
                "detections": detections
            })

        except Exception as e:
            print(f"❌ Error processing {obj_id}: {e}")
            predictions.append({"video_id": obj_id, "detections": []})

    # ===============================================================
    # 4️⃣ LƯU FILE JSON
    # ===============================================================
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\n✅ Tracking completed.")
    print(f"📄 JSON saved to: {output_file}")
    if save_debug_video:
        print(f"🎥 Debug videos saved to: {debug_video_dir}")
