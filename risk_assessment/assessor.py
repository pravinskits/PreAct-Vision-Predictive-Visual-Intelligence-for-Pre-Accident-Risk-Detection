class RiskAssessor:
    def __init__(self, high_risk_ttc=25.0, medium_risk_ttc=60.0, speed_threshold=5.0):
        """
        Initialize the rule-based proactive risk scoring engine.
        Evaluates TTC, proximity to vehicle bumper, closing speed, and road user class.
        """
        self.high_risk_ttc = high_risk_ttc
        self.medium_risk_ttc = medium_risk_ttc
        self.speed_threshold = speed_threshold

    def assess(self, predicted_objects):
        """
        Assess risk for predicted objects based on their trajectories, Time-to-Collision (TTC),
        and proximity to the ego vehicle.
        """
        risk_alerts = {}
        for obj in predicted_objects:
            obj_id = obj['id']
            ttc = obj.get('ttc', float('inf'))
            vx, vy = obj['velocity']
            dist_to_ego = obj.get('dist_to_ego', 999.0)
            cls_name = obj.get('class_name', 'Object').lower()
            speed = (vx**2 + vy**2)**0.5
            
            # Rule 1: High TTC collision trajectory
            if ttc <= self.high_risk_ttc:
                risk_level = 'High'
            # Rule 2: Immediate proximity hazard (very close to vehicle front)
            elif dist_to_ego < 110:
                risk_level = 'High'
            # Rule 3: Vulnerable road users (pedestrians / bikes in travel zone)
            elif cls_name in ['person', 'bicycle'] and dist_to_ego < 200:
                risk_level = 'High' if dist_to_ego < 130 or speed > 1.5 else 'Medium'
            # Rule 4: Approaching vehicle with medium TTC
            elif ttc <= self.medium_risk_ttc:
                risk_level = 'Medium'
            # Rule 5: Vehicles in nearby lane corridor or high lateral crossing speed
            elif dist_to_ego < 190 or (abs(vx) > 2.5 and dist_to_ego < 250):
                risk_level = 'Medium'
            # Rule 6: High approach speed
            elif speed >= self.speed_threshold and vy > 0.8:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
                
            risk_alerts[obj_id] = risk_level
            
        return risk_alerts
