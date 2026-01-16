# inferences.py
import os
import json
import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from super_gradients.training import models
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ReferenceGuidedInference:
    def __init__(
        self,
        model_path: str,
        similarity_threshold: float = 0.4,
        device: torch.device | None = None,
        score_threshold: float = 0.01,
        save_debug_video: bool = False
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold
        self.save_debug_video = save_debug_video

        logging.info(f"Loading detection model from {model_path} on device {self.device}")
        # load detection model (YOLO-NAS)
        self.detector = models.get(
            "yolo_nas_s",
            num_classes=1,
            checkpoint_path=model_path
        )
        self.detector.to(self.device)
        self.detector.eval()

        # Post-processing with low score threshold to capture potential objects
        self.post_processing = PPYoloEPostPredictionCallback(
            score_threshold=score_threshold,
            nms_threshold=0.5,
            nms_top_k=1000,
            max_predictions=300
        )

        # Feature extractor (use backbone) and transforms
        self.feature_extractor = self._create_feature_extractor()
        # Ensure Resize before ToTensor
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        logging.info(f"Reference-guided inference initialized on {self.device} with similarity_threshold={self.similarity_threshold}")

    def _create_feature_extractor(self):
        """Use backbone as feature extractor"""
        return self.detector.backbone

    def extract_features(self, image: np.ndarray):
        """Extract L2-normalized feature vector (cpu numpy) from image (RGB numpy HWC)."""
        if not isinstance(image, np.ndarray):
            raise ValueError("extract_features expects numpy image (HWC, RGB)")

        # transform -> tensor (1, C, H, W)
        img_t = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(img_t)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            features = F.normalize(features, p=2, dim=1)

            feat_cpu = features.cpu().numpy()

        return feat_cpu

    def load_reference_features(self, reference_dir: str):
        """Load reference images, extract features, and return normalized average vector."""
        reference_dir = Path(reference_dir)
        if not reference_dir.exists():
            logging.warning(f"Reference dir not found: {reference_dir}")
            return None

        ref_images = []
        for img_path in sorted(reference_dir.glob("*.*")):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(img_path))
                if img is None:
                    logging.warning(f"Failed to read ref image: {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ref_images.append(img)

        if not ref_images:
            logging.warning(f"No reference images found in {reference_dir}")
            return None

        # Extract features
        ref_feats = []
        pbar = tqdm(ref_images, desc="Extracting reference features", unit="image", leave=False)
        for img in pbar:
            try:
                f = self.extract_features(img)
                ref_feats.append(f)
            except Exception as e:
                logging.warning(f"Reference feature extraction failed for one image: {e}")
        pbar.close()

        if not ref_feats:
            logging.warning("No reference features extracted successfully")
            return None

        ref_feats = np.concatenate(ref_feats, axis=0)
        avg = np.mean(ref_feats, axis=0, keepdims=True)
        norm = np.linalg.norm(avg)
        if norm == 0:
            logging.warning("Reference average feature norm is zero")
            return None
        avg = avg / norm
        return avg

    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray):
        """Cosine similarity between two 1-D vectors (both numpy)"""
        return float(np.dot(features1, features2.T).item())

    def _preprocess_frame(self, frame: np.ndarray, input_size: int = 640):
        """Prepare frame for detector: resize with letterbox/pad to input_size, return tensor on device."""
        h, w = frame.shape[:2]
        scale = min(input_size / w, input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        padded = padded.astype(np.float32) / 255.0
        padded = padded.transpose(2, 0, 1)
        tensor = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        return tensor

    def _parse_predictions(self, processed_predictions):
        """
        Robustly parse processed_predictions result of post_processing.
        Returns a list (per-image) of bboxes numpy: N x >=6 (x1,y1,x2,y2,conf,cls)
        """
        if processed_predictions is None:
            return []

        if isinstance(processed_predictions, (list, tuple)):
            # expected list per image
            out = []
            for p in processed_predictions:
                if isinstance(p, dict) and "bboxes" in p:
                    out.append(np.array(p["bboxes"]))
                elif hasattr(p, "bboxes"):
                    try:
                        out.append(p.bboxes.cpu().numpy())
                    except Exception:
                        out.append(np.array([]))
                elif isinstance(p, torch.Tensor):
                    out.append(p.cpu().numpy())
                else:
                    out.append(np.array([]))
            return out

        # fallback single image
        p = processed_predictions
        if isinstance(p, dict) and "bboxes" in p:
            return [np.array(p["bboxes"])]
        if hasattr(p, "bboxes"):
            try:
                return [p.bboxes.cpu().numpy()]
            except Exception:
                return [np.array([])]
        if isinstance(p, torch.Tensor):
            return [p.cpu().numpy()]
        return []

    def _filter_detections_by_similarity(self, predictions, frame, ref_features, frame_number):
        """Filter detections based on cosine similarity with reference features."""
        filtered = []
        # predictions: expected numpy array N x >=6 with x1,y1,x2,y2,conf,cls
        if predictions is None or len(predictions) == 0:
            return filtered

        # Ensure numpy
        bboxes = np.array(predictions)
        h, w = frame.shape[:2]

        raw_count = bboxes.shape[0]
        kept = 0

        for idx in range(bboxes.shape[0]):
            bbox = bboxes[idx]
            if bbox.size < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = bbox[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            try:
                det_feat = self.extract_features(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                sim = self.compute_similarity(det_feat[0], ref_features[0])
                logging.debug(f"Frame {frame_number} bbox#{idx} conf={conf:.3f} sim={sim:.3f}")
                if sim >= self.similarity_threshold:
                    filtered.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(conf), "sim": float(sim)})
                    kept += 1
            except Exception as e:
                logging.warning(f"Feature extraction failed on frame {frame_number} bbox#{idx}: {e}")
                continue

        logging.debug(f"Frame {frame_number}: raw_bboxes={raw_count} kept={kept}")
        return filtered

    def _create_detection_groups(self, frame_detections):
        """
        Group consecutive frames containing detections into intervals.
        Output: list of {"bboxes": [ {frame, x1,y1,x2,y2}, ... ] }
        """
        if not frame_detections:
            return []

        sorted_frames = sorted(frame_detections.keys())
        intervals = []
        current = []

        for f in sorted_frames:
            if not current:
                current = [f]
            else:
                if f == current[-1] + 1:
                    current.append(f)
                else:
                    intervals.append(current)
                    current = [f]
        if current:
            intervals.append(current)

        groups = []
        for interval in intervals:
            bboxes_list = []
            for fr in interval:
                for bbox in frame_detections[fr]:
                    bboxes_list.append({
                        "frame": fr,
                        "x1": int(bbox["x1"]),
                        "y1": int(bbox["y1"]),
                        "x2": int(bbox["x2"]),
                        "y2": int(bbox["y2"])
                    })
            if bboxes_list:
                groups.append({"bboxes": bboxes_list})
        return groups

    def process_video(self, video_path: str, reference_dir: str, max_frames: int | None = None):
        """Process single video using reference-guided detection. Returns dict for submission."""
        video_id = Path(video_path).parent.name
        logging.info(f"Processing video: {video_id} with reference: {reference_dir}")

        start_time = time.time()
        ref_features = self.load_reference_features(reference_dir)
        if ref_features is None:
            logging.warning(f"No reference features for {video_id}. Skipping.")
            return {"video_id": video_id, "detections": []}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video: {video_path}")
            return {"video_id": video_id, "detections": []}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25.0
        logging.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")

        frame_detections = {}
        frame_idx = 0

        # Setup debug video writer if requested
        debug_writer = None
        if self.save_debug_video:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            debug_out_path = Path(video_path).with_name(f"{video_id}_debug.mp4")
            debug_writer = cv2.VideoWriter(str(debug_out_path), fourcc, fps, (width, height))
            logging.info(f"Debug video will be saved to {debug_out_path}")

        pbar = tqdm(total=total_frames if max_frames is None else min(total_frames, max_frames),
                    desc=f"Processing {video_id}", unit="frame")
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self._preprocess_frame(frame_rgb)

            with torch.no_grad():
                try:
                    predictions = self.detector(input_tensor)
                    processed = self.post_processing(predictions)
                except Exception as e:
                    logging.warning(f"Detection failed on frame {frame_idx}: {e}")
                    processed = None

            parsed = self._parse_predictions(processed)
            raw_bboxes = parsed[0] if parsed else np.array([])

            logging.debug(f"Frame {frame_idx}: raw detections = {len(raw_bboxes) if isinstance(raw_bboxes, np.ndarray) else 0}")

            detections_this_frame = self._filter_detections_by_similarity(
                raw_bboxes, frame_rgb, ref_features, frame_idx
            )

            if detections_this_frame:
                frame_detections[frame_idx] = detections_this_frame

            # Draw debug boxes if enabled
            if self.save_debug_video:
                vis = frame.copy()
                for d in detections_this_frame:
                    cv2.rectangle(vis, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 255, 0), 2)
                    cv2.putText(vis, f"{d.get('sim',0):.2f}", (d["x1"], max(0, d["y1"]-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                debug_writer.write(vis)

            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix({
                "detections": len(detections_this_frame),
                "total_detections": sum(len(v) for v in frame_detections.values())
            })

        cap.release()
        pbar.close()
        if debug_writer:
            debug_writer.release()

        detection_groups = self._create_detection_groups(frame_detections)
        end_time = time.time()
        logging.info(f"Completed {video_id}: {len(detection_groups)} detection groups in {end_time - start_time:.2f}s")

        return {"video_id": video_id, "detections": detection_groups}

    def process_all_videos(self, test_samples_dir: str, model_path: str, output_json_path: str,
                           similarity_threshold: float = None, save_debug_videos: bool = False, max_frames_per_video: int | None = None):
        """Convenience wrapper to process all video directories and write submission JSON."""
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold

        self.save_debug_video = save_debug_videos

        test_dir = Path(test_samples_dir)
        video_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if not video_dirs:
            logging.error(f"No video directories found in {test_samples_dir}")
            return

        submission = []
        pbar_v = tqdm(sorted(video_dirs), desc="Processing videos", unit="video")
        for vd in pbar_v:
            video_path = vd / "drone_video.mp4"
            reference_dir = vd / "object_images"
            pbar_v.set_postfix({"video": vd.name})

            if not video_path.exists():
                logging.warning(f"Video not found: {video_path}")
                submission.append({"video_id": vd.name, "detections": []})
                continue

            if not reference_dir.exists():
                logging.warning(f"Reference images not found: {reference_dir}")
                submission.append({"video_id": vd.name, "detections": []})
                continue

            try:
                result = self.process_video(str(video_path), str(reference_dir), max_frames=max_frames_per_video)
                submission.append(result)
            except Exception as e:
                logging.error(f"Error processing {vd.name}: {e}")
                submission.append({"video_id": vd.name, "detections": []})

        # Write JSON
        try:
            with open(output_json_path, 'w') as f:
                json.dump(submission, f, indent=2)
            logging.info(f"Submission file saved: {output_json_path}")
        except Exception as e:
            logging.error(f"Failed to write submission file {output_json_path}: {e}")

        return submission
