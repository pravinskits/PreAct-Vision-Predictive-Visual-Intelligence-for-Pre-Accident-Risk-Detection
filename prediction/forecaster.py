class TrajectoryForecaster:
    def __init__(self, future_frames=30):
        """
        Initialize the trajectory forecaster.
        """
        self.future_frames = future_frames
        self.frame_height = 360  # Default normalized height
        self.frame_width = 640

    def forecast(self, tracked_objects):
        """
        Predict future trajectories for tracked objects and calculate Time-to-Collision & Proximity.
        """
        predicted_objects = []
        for obj in tracked_objects:
            cx, cy = obj['centroid']
            vx, vy = obj['velocity']
            
            # 1. Predict future trajectory points
            future_points = []
            for i in range(1, self.future_frames + 1):
                px = int(cx + vx * i)
                py = int(cy + vy * i)
                future_points.append((px, py))
                
            obj['future_trajectory'] = future_points
            
            # 2. Distance to ego vehicle bumper & central corridor
            distance_to_bottom = max(0, self.frame_height - cy)
            dist_to_ego = ((cx - self.frame_width / 2.0)**2 + (self.frame_height - cy)**2)**0.5
            obj['dist_to_ego'] = dist_to_ego
            
            # 3. TTC Calculation (Time to Collision)
            ttc = float('inf')
            if vy > 0.1: # Object moving towards ego vehicle
                ttc = distance_to_bottom / max(0.1, vy)
            elif distance_to_bottom < 90 and abs(cx - self.frame_width / 2.0) < 140:
                # Stopped or slow obstacle directly in front of vehicle bumper
                ttc = 6.0
            elif distance_to_bottom < 160 and abs(cx - self.frame_width / 2.0) < 120:
                # Immediate driving corridor hazard
                ttc = 20.0
            
            obj['ttc'] = ttc
            predicted_objects.append(obj)
            
        return predicted_objects
