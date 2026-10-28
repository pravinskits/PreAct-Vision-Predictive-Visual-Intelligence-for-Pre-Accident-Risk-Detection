import cv2

class Visualizer:
    def __init__(self):
        """
        Initialize the visualizer for rendering bounding boxes, 
        trajectories, and risk warnings.
        """
        self.risk_colors = {
            'High': (0, 0, 255),    # Red
            'Medium': (0, 255, 255),# Yellow
            'Low': (0, 255, 0),     # Green
            'None': (255, 255, 255) # White
        }

    def draw_annotations(self, frame, predicted_objects, risk_alerts, fps=None, frame_idx=None):
        """
        Draw predictive risk alerts, tracking data, and FPS overlay.
        """
        annotated_frame = frame.copy()
        
        for obj in predicted_objects:
            obj_id = obj['id']
            x1, y1, x2, y2 = obj['bbox']
            risk_level = risk_alerts.get(obj_id, 'None')
            color = self.risk_colors.get(risk_level, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID, Class Name, Risk Level, and TTC/Velocity
            vx, vy = obj['velocity']
            speed = (vx**2 + vy**2)**0.5
            cls_name = obj.get('class_name', 'Object')
            ttc = obj.get('ttc', float('inf'))
            if ttc != float('inf') and ttc < 90:
                label = f"#{obj_id} {cls_name} [{risk_level}] TTC:{ttc:.1f}"
            else:
                label = f"#{obj_id} {cls_name} [{risk_level}] Spd:{speed:.1f}"
            
            # Dynamic text background to prevent smearing
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            pad = 5
            box_x = x1
            box_y = y1 - lh - 2*pad - 5
            
            # Draw exact background box
            cv2.rectangle(annotated_frame, (box_x, box_y), (box_x + lw + 2*pad, box_y + lh + 2*pad), (0, 0, 0), -1)
            
            # Text origin relative to the top_left box coordinates
            text_org = (box_x + pad, box_y + lh + pad)
            cv2.putText(annotated_frame, label, text_org, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
            # Draw trailing history
            history = obj.get('history', [])
            if len(history) > 1:
                for i in range(1, len(history)):
                    cv2.line(annotated_frame, history[i-1], history[i], color, 2)
                    
            # Draw forecasted trajectory as a vector arrow
            future_trajectory = obj.get('future_trajectory', [])
            if len(future_trajectory) > 5:
                start_pt = future_trajectory[0]
                end_pt = future_trajectory[-1] # Arrow pointing to the end of the 30-frame prediction
                cv2.arrowedLine(annotated_frame, start_pt, end_pt, color, 2, tipLength=0.1)

        # HUD overlay: Frame index and live FPS badge
        hud_parts = []
        if frame_idx is not None:
            hud_parts.append(f"FRAME #{frame_idx}")
        if fps is not None:
            hud_parts.append(f"{fps:.1f} FPS")
        if hud_parts:
            hud_text = " | ".join(hud_parts)
            (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (10, 10), (10 + hw + 12, 10 + hh + 12), (15, 15, 15), -1)
            cv2.rectangle(annotated_frame, (10, 10), (10 + hw + 12, 10 + hh + 12), (0, 230, 118), 1)
            cv2.putText(annotated_frame, hud_text, (16, 10 + hh + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 118), 1, cv2.LINE_AA)

        return annotated_frame
