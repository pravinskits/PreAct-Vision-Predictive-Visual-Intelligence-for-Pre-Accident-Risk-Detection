import torch
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    # Road traffic classes in COCO:
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 6: train, 7: truck
    # Overhead/aerial surveillance misclassifications:
    # 8: boat, 28: suitcase, 67: cell phone (vehicles frequently misclassified from bird's-eye view)
    CANDIDATE_CLASSES = [0, 1, 2, 3, 5, 6, 7, 8, 28, 67]
    
    CLASS_NAMES = {
        0: 'Person',
        1: 'Bicycle',
        2: 'Car',
        3: 'Motorcycle',
        5: 'Bus',
        6: 'Train',
        7: 'Truck'
    }

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.15):
        """
        Initialize the YOLO object detector for road traffic hazard detection.
        High recall default threshold (0.15) captures distant and partially occluded objects.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        print(f"ObjectDetector initialized on device: {self.device} with conf_threshold: {self.conf_threshold}")

    @staticmethod
    def _compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def detect(self, frame, imgsz=640):
        """
        Detect road hazards (cars, pedestrians, bicycles, buses, trucks, motorcycles).
        Includes aerial/drone surveillance vehicle recovery and IoU deduplication.
        """
        results = self.model(
            frame,
            imgsz=imgsz,
            conf=self.conf_threshold,
            classes=self.CANDIDATE_CLASSES,
            verbose=False
        )
        
        raw_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                
                # Aerial / CCTV top-down vehicle remapping:
                # Cars viewed from top-down often look like rectangular rounded blocks (cell phone, suitcase, boat).
                # If large enough (>40px in width and height), remap to Car. Otherwise ignore spurious items.
                if cls_id in [8, 28, 67]:
                    if w >= 40 and h >= 40:
                        cls_id = 2
                        cls_name = 'Car'
                    else:
                        continue
                else:
                    cls_name = self.CLASS_NAMES.get(cls_id, self.model.names.get(cls_id, 'Vehicle'))

                raw_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': conf
                })

        # Sort detections by confidence descending for Non-Maximum Suppression
        raw_detections.sort(key=lambda d: d['confidence'], reverse=True)

        # Deduplicate overlapping detections (e.g. Car vs Bus on the exact same vehicle)
        filtered_detections = []
        for det in raw_detections:
            keep = True
            for accepted in filtered_detections:
                iou = self._compute_iou(det['bbox'], accepted['bbox'])
                if iou > 0.50:
                    keep = False
                    break
            if keep:
                filtered_detections.append(det)

        return filtered_detections
