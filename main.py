import cv2
import time
import argparse
import sys
import os

try:
    from perception.detector import ObjectDetector
    from tracking.tracker import MultiObjectTracker
    from prediction.forecaster import TrajectoryForecaster
    from risk_assessment.assessor import RiskAssessor
    from utils.visualizer import Visualizer
except ImportError as e:
    print(f"[ERROR] Failed to import project modules: {e}")
    print("Please ensure you have installed all dependencies via 'pip install -r requirements.txt'")
    sys.exit(1)

def main(video_path, output_path):
    print("Initializing PreAct-Vision Modules...")
    try:
        detector = ObjectDetector()
        tracker = MultiObjectTracker()
        forecaster = TrajectoryForecaster()
        assessor = RiskAssessor()
        visualizer = Visualizer()
    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline modules. {e}")
        sys.exit(1)

    # Handle webcam or check file existence
    if str(video_path).isdigit():
        video_path = int(video_path)
    elif not os.path.exists(video_path):
        print(f"[ERROR] Input video file not found at path: '{video_path}'")
        print("Please verify the file path and try again.")
        sys.exit(1)

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream or file at '{video_path}'. The file might be corrupted or in an unsupported format.")
        sys.exit(1)

    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_prop = cap.get(cv2.CAP_PROP_FPS)
    if fps_prop == 0 or fps_prop != fps_prop:
        fps_prop = 30.0 # fallback

    forecaster.frame_height = height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_prop, (width, height))

    print("Starting processing pipeline...")
    frame_count = 0
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = frame.copy()

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        try:
            # 1. Perception
            detections = detector.detect(frame)

            # 2. Tracking
            tracked_objects = tracker.update(detections, frame)

            # 3. Prediction
            predicted_objects = forecaster.forecast(tracked_objects)

            # 4. Risk Assessment
            risk_alerts = assessor.assess(predicted_objects)
        except Exception as e:
            print(f"[ERROR] Pipeline execution crashed at frame {frame_count}: {e}")
            break

        # Logging Hazard Alerts to console
        for obj_id, risk_level in risk_alerts.items():
            if risk_level == 'High':
                print(f"[WARNING] High collision risk predicted for Object #{obj_id} at frame {frame_count}!")
            elif risk_level == 'Medium':
                print(f"[CAUTION] Medium hazard risk tracked for Object #{obj_id} at frame {frame_count}.")

        # 5. Visualization with HUD overlay
        annotated_frame = visualizer.draw_annotations(frame, predicted_objects, risk_alerts, fps=fps, frame_idx=frame_count + 1)

        # Write output frame
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames... (Current FPS: {fps:.1f})")

    cap.release()
    out.release()
    print(f"Processing complete! Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PreAct-Vision Predictive Hazard Detection Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file or 0 for webcam")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output annotated video")
    
    args = parser.parse_args()
    
    try:
        main(args.input, args.output)
    except KeyboardInterrupt:
        print("\n[INFO] Processing gracefully interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {e}")
        sys.exit(1)
