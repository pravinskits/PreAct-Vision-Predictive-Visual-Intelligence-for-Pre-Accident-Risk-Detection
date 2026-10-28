import numpy as np

class MultiObjectTracker:
    def __init__(self, max_disappeared=40, max_distance=150):
        """
        Initialize a robust multi-object tracker with velocity estimation and dead reckoning.
        Prevents flickering when detections are momentarily dropped or when running alternate frames.
        """
        self.next_object_id = 0
        self.objects = {} # id: {'bbox': (x1,y1,x2,y2), 'centroid': (cx,cy), 'velocity': (vx,vy), 'class_id': id, 'class_name': name}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, detection, centroid):
        self.objects[self.next_object_id] = {
            'bbox': detection['bbox'],
            'centroid': centroid,
            'velocity': (0.0, 0.0),
            'history': [centroid],
            'class_id': detection.get('class_id', 0),
            'class_name': detection.get('class_name', 'Object')
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def _extrapolate(self, object_id):
        """
        Apply dead reckoning: advance object position using last estimated velocity.
        """
        if object_id not in self.objects:
            return
        cx, cy = self.objects[object_id]['centroid']
        vx, vy = self.objects[object_id]['velocity']
        x1, y1, x2, y2 = self.objects[object_id]['bbox']
        
        new_cx = int(cx + vx)
        new_cy = int(cy + vy)
        new_bbox = (int(x1 + vx), int(y1 + vy), int(x2 + vx), int(y2 + vy))
        
        self.objects[object_id]['centroid'] = (new_cx, new_cy)
        self.objects[object_id]['bbox'] = new_bbox
        self.objects[object_id]['history'].append((new_cx, new_cy))
        if len(self.objects[object_id]['history']) > 25:
            self.objects[object_id]['history'].pop(0)

    def update(self, detections, frame=None):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] <= 5:
                    self._extrapolate(object_id)
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._get_tracked_objects()

        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(detections[i], input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            
            # Predict each object's expected centroid using its current velocity
            projected_centroids = []
            for obj_id in object_ids:
                cx, cy = self.objects[obj_id]['centroid']
                vx, vy = self.objects[obj_id]['velocity']
                projected_centroids.append((cx + int(vx), cy + int(vy)))

            # Compute Euclidean distance between projected centroids and new detections
            D = np.linalg.norm(np.array(projected_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                prev_centroid = self.objects[object_id]['centroid']
                new_centroid = input_centroids[col]
                
                vx = new_centroid[0] - prev_centroid[0]
                vy = new_centroid[1] - prev_centroid[1]
                
                # Exponential moving average for velocity to smooth jitter
                prev_vx, prev_vy = self.objects[object_id]['velocity']
                alpha = 0.5
                vx = alpha * vx + (1 - alpha) * prev_vx
                vy = alpha * vy + (1 - alpha) * prev_vy

                self.objects[object_id]['centroid'] = new_centroid
                self.objects[object_id]['bbox'] = detections[col]['bbox']
                self.objects[object_id]['velocity'] = (vx, vy)
                self.objects[object_id]['history'].append(new_centroid)
                
                # Retain or upgrade class name if new detection is more specific
                det_cls_name = detections[col].get('class_name')
                if det_cls_name and det_cls_name != 'Object':
                    self.objects[object_id]['class_name'] = det_cls_name
                    self.objects[object_id]['class_id'] = detections[col].get('class_id', self.objects[object_id]['class_id'])
                
                if len(self.objects[object_id]['history']) > 25:
                    self.objects[object_id]['history'].pop(0)
                
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] <= 5:
                    self._extrapolate(object_id)
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(detections[col], input_centroids[col])

        return self._get_tracked_objects()
        
    def _get_tracked_objects(self):
        tracked = []
        for obj_id, data in self.objects.items():
            # Don't display ghosts that have been missing for more than 5 consecutive frames
            if self.disappeared.get(obj_id, 0) > 5:
                continue
            tracked.append({
                'id': obj_id,
                'bbox': data['bbox'],
                'centroid': data['centroid'],
                'velocity': data['velocity'],
                'history': data['history'],
                'class_id': data['class_id'],
                'class_name': data.get('class_name', 'Object')
            })
        return tracked

