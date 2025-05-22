import os
import json
import csv
import tempfile
from pathlib import Path
from collections import defaultdict, deque
import random

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from deep_sort_realtime.deepsort_tracker import DeepSort


class TrafficAnalyzer:
    """Main class for traffic analysis with vehicle tracking and turn detection"""
    
    def __init__(self):
        self.setup_cuda()
        self.setup_models()
        self.setup_colors()
        self.setup_tracking_data()

    def setup_cuda(self):
        """Setup CUDA device and check availability because use of GPU CUDA can give a speedup"""
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            self.device = torch.device('cuda')
            # Print GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"CUDA available: Using GPU - {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            
            # Frees unused GPU memory back to the system useful to avoid out-of-memory errors.
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')
            print("CUDA not available: Using CPU")
            print("Note: For better performance, consider using a CUDA-enabled environment")
    
    
    def setup_models(self):
        """Initialize YOLO model and DeepSort tracker
                - Chosen specifically for vehicle detection happens to be YOLOv8m as it balances speed with accuracy.
                - Confidence threshold 0.35: Captures vehicles with minimization of false positives
                - IOU threshold 0.45 reduces detections that duplicate within scenes that are crowded.
                - DeepSORT keeps vehicle IDs through frames then is used to track many objects and also has Appearance Features.
        """
        # Use medium-sized YOLO model for better accuracy
        # Alternatives: yolov8n (faster), yolov8l/x (more accurate but slower)
        self.model = YOLO("yolov8m.pt")

        # gpu(cuda) for speed up if available or cpu
        if self.cuda_available:
            self.model.to(self.device)
            print("YOLO model loaded on GPU")
        else:
            print("YOLO model loaded on CPU")
            
        # Confidence threshold: balances false positives/negatives    
        self.model.conf = 0.35 
        # IOU threshold for NMS: standard value to remove duplicate boxes 
        self.model.iou = 0.45  
        
        # Only detect car using COCO class IDs
        self.CAR_CLASSES = [2]  
        
        # Enhanced tracker settings
        self.tracker = DeepSort(
            max_age=50,# Frames to keep track after last detection
            n_init=3,# Number of detections before track is confirmed
            max_iou_distance=0.7,# Maximum IOU distance for track association
            max_cosine_distance=0.3,# Maximum cosine distance for appearance matching
            nn_budget=120# Maximum samples stored for appearance matching
        )
    
    def setup_colors(self):
        """Configure color"""
        # Single color for all turn types (BGR format)
        self.SINGLE_TURN_COLOR = (0, 255, 0)  # Green
        
        # Color mapping for different turn types
        self.turning_bb_color = {
            'right': self.SINGLE_TURN_COLOR,
            'left': self.SINGLE_TURN_COLOR,
            'u-turn': self.SINGLE_TURN_COLOR,
            'straight': self.SINGLE_TURN_COLOR,
            'unknown': (128, 128, 128)  # Gray for unknown turns
        }
    
    def setup_tracking_data(self):
        """Initialize tracking data structures for vehicle trajectories and turn statistics"""
        self.trajectories = {}
        self.car_turns = {}
        self.turn_counts = {"left": 0, "right": 0, "straight": 0, "u-turn": 0, "unknown": 0}
        self.total_cars = 0
        self.previous_tracks = set()
    
    def define_intersection_zones(self, width, height):
        """Define entry and exit zones for intersection analysis"""
        # Entry zones - where vehicles enter the intersection
        entry_zones = {
            "north_in": [[768.82, 333.13], [815.88, 368.43], [994.31, 137.05], [927.64, 115.49]],
            "south_in": [[982.54, 988.03], [1208.03, 737.05], [1245.29, 784.11], [1059.01, 1003.72]],
            "east_in": [[1231.56, 348.82], [1270.78, 309.60], [1482.54, 491.96], [1470.78, 550.78]],
            "west_in": [[796.27, 793.92], [560.98, 650.78], [547.25, 709.60], [747.25, 833.13]]
        }
        
        # Exit zones - where vehicles exit the intersection
        exit_zones = {
            "north_out": [[562.94, 621.37], [796.27, 388.03], [749.21, 352.74], [555.09, 552.74]],
            "south_out": [[1229.60, 717.45], [1472.74, 531.17], [1484.50, 615.49], [1280.58, 758.62]],
            "east_out": [[1204.11, 323.33], [1006.07, 140.98], [1076.66, 117.45], [1239.41, 286.07]],
            "west_out": [[774.70, 839.01], [811.96, 803.72], [1008.03, 995.88], [919.80, 997.84]]
        }
        
        # Convert to Polygon objects
        zones = {}
        for name, coords in {**entry_zones, **exit_zones}.items():
            zones[name] = Polygon(coords)
        
        return zones
    
    def get_zone_turn_mapping(self):
            """Map entry-exit zone combinations to turn directions.Explicit mapping for each possible movement through the intersection"""        
            return {
            # From North Entry
            ("north_in", "east_out"): "right",
            ("north_in", "south_out"): "straight",
            ("north_in", "west_out"): "left",
            ("north_in", "north_out"): "u-turn",
            
            # From South Entry
            ("south_in", "west_out"): "right",
            ("south_in", "north_out"): "straight",
            ("south_in", "east_out"): "left",
            ("south_in", "south_out"): "u-turn",
            
            # From East Entry
            ("east_in", "south_out"): "right",
            ("east_in", "west_out"): "straight",
            ("east_in", "north_out"): "left",
            ("east_in", "east_out"): "u-turn",
            
            # From West Entry
            ("west_in", "north_out"): "right",
            ("west_in", "east_out"): "straight",
            ("west_in", "south_out"): "left",
            ("west_in", "west_out"): "u-turn",
        }
    
    def find_zone(self, cx, cy, zones):
        """
            Find which zone contains the given point.
            Uses Shapely's efficient point-in-polygon algorithm.
                1.cx, cy: Point coordinates (vehicle center)
                2.zones: Dictionary of zone_name -> Polygon mappings
            Zone name if point is inside a zone,or None is returned
        """
        point = Point(cx, cy)
        for zone_name, polygon in zones.items():
            if polygon.contains(point):
                return zone_name
        return None
    
    def calculate_angle_between_vectors(self, v1, v2):
        """Calculate signed angle between two vectors in degrees
        Decision: Angle is used for turn classification (left, right, straight, u-turn).
        """
        # Normalize vectors to unit length to ensure consistent angle calculation
        v1_norm = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
        v2_norm = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
        
        # Calculate dot product and keep in valid range
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        
        # Calculate signed angle
        det = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
        angle = np.arctan2(det, dot)
        return np.degrees(angle)
    
    def analyze_trajectory(self, trajectory, zones):
        """
        Analyze vehicle trajectory to determine turn direction.
        
        Multi-modal Analysis Approach:
        1. Primary: Angle-based analysis (more robust to zone boundary issues)
        2. Fallback: Zone-based analysis using entry/exit zone mapping
        3. Minimum trajectory length ensures reliable classification
        
        The angle-based method is prioritized as it's more robust to
        zone boundary (as it is labelling is specific to this video).
        """
        if len(trajectory) < 20:
            return {"entry_zone": None, "exit_zone": None, "turn_type": None, "valid": False}
        
        # Find entry and exit zones
        entry_zone = self._find_entry_zone(trajectory[:10], zones)#First 10 points
        exit_zone = self._find_exit_zone(trajectory[-10:], zones)#Last 10 points
        
        # Use angle-based detection for turn classification
        turn_type = self._classify_turn_by_angle(trajectory)
        classification_method = "angle"


        # Fallback: If angle-based fails, use zone-based classification
        if turn_type is None and entry_zone and exit_zone:
            zone_mapping = self.get_zone_turn_mapping()
            turn_type = zone_mapping.get((entry_zone, exit_zone))
            classification_method = "zone" if turn_type else None

        
        return {
            "entry_zone": entry_zone,
            "exit_zone": exit_zone,
            "turn_type": turn_type,
            "valid": True,
            "angle": getattr(self, '_last_angle', None),
            "classification_method": classification_method

        }
    
    def _find_entry_zone(self, start_points, zones):
        """Find entry zone from starting points"""
        for pt in start_points:
            zone = self.find_zone(*pt, zones)
            if zone and zone.endswith('_in'):# Only consider entry zones
                return zone
        return None
    
    def _find_exit_zone(self, end_points, zones):
        """Find exit zone from ending points"""
        for pt in end_points:
            zone = self.find_zone(*pt, zones)
            if zone and zone.endswith('_out'): # Only consider exit zones
                return zone
        return None
    
    def _classify_turn_by_angle(self, trajectory):
        """
        Classify turn based on trajectory angle analysis.
        
        Algorithm:
        1. Calculate initial direction vector (first 5 points)
        2. Calculate final direction vector (last 5 points)  
        3. Compute angle between vectors
        4. Classify based on empirically-determined thresholds
        
        Angle Thresholds (in degrees):
        - Straight: -30 to +30 
        - Right turn: +30 to +120 (clockwise rotation)
        - Left turn: -120 to -30 (counter-clockwise rotation)
        - U-turn: >+120 or <-120 (large angle changes)
        
        """
        if len(trajectory) < 10:
            return None
        
        # Calculate direction vectors
        start_pts = trajectory[:5] # Initial direction
        end_pts = trajectory[-5:] # Final direction
        
        start_vec = [start_pts[-1][0] - start_pts[0][0], start_pts[-1][1] - start_pts[0][1]]
        end_vec = [end_pts[-1][0] - end_pts[0][0], end_pts[-1][1] - end_pts[0][1]]
        
        if np.linalg.norm(start_vec) > 0 and np.linalg.norm(end_vec) > 0:
            angle = self.calculate_angle_between_vectors(start_vec, end_vec)
            self._last_angle = angle
            
            # Classify based on angle thresholds
            if -30 <= angle <= 30:
                return "straight"
            elif 30 < angle <= 120:
                return "right"
            elif -120 <= angle < -30:
                return "left"
            else:
                return "u-turn"
        
        return None


class VideoProcessor:
    """Handles video processing and visualization"""
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def get_box_details(self, boxes):
        """Extract box details from YOLO results"""
        return boxes.cls, boxes.xyxy, boxes.conf, boxes.xywh
    
    def draw_label(self, img, text, pos, bg_color, text_color=(255, 255, 255)):
        """Draw text label with background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        end_x = pos[0] + size[0] + 2
        end_y = pos[1] - size[1] - 2
        
        cv2.rectangle(img, pos, (end_x, end_y), bg_color, -1)
        cv2.putText(img, text, (pos[0], pos[1] - 2), font, font_scale, text_color, thickness)
    
    def draw_enhanced_bbox(self, frame, x1, y1, x2, y2, track_id, turn_type=None):
        """Draw enhanced bounding box with corner markers"""
        # Get color and line thickness
        if turn_type and turn_type in self.analyzer.turning_bb_color:
            bb_color = self.analyzer.turning_bb_color[turn_type]
            line_thickness = 3
        else:
            bb_color = (0, 255, 0)
            line_thickness = 2
        
        # Draw main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), bb_color, line_thickness)
        
        # Draw corner markers
        self._draw_corner_markers(frame, x1, y1, x2, y2, bb_color)
        
        # Draw ID 
        label_text = f"ID:{track_id}"
        label_bg_color = bb_color if turn_type else (0, 0, 0)
        self.draw_label(frame, label_text, (x1, y1-10), label_bg_color)
    
    def _draw_corner_markers(self, frame, x1, y1, x2, y2, color):
        """Draw corner markers for bounding box"""
        corner_length = 15
        corner_thickness = 3
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
    
    def draw_polygon_zones(self, frame, zones):
        """Draw zone polygons with minimal visibility"""
        for zone_name, polygon in zones.items():
            pts = np.array(polygon.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(50, 50, 50), thickness=1)
    
    def draw_debug_info(self, frame, total_detected, valid_tracked, turn_counts, total_cars):
        """Draw debug information overlay"""
        h, w = frame.shape[:2]
        
        # Main stats box
        cv2.rectangle(frame, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 180), (255, 255, 255), 2)
        
        # Detection stats
        stats = [
            f"Detected this frame: {total_detected}",
            f"Actively tracked: {valid_tracked}",
            f"Total vehicles: {total_cars}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Turn counts
        y_offset = 105
        cv2.putText(frame, "Turn Counts:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        for turn_type, count in turn_counts.items():
            if turn_type in self.analyzer.turning_bb_color:
                color = self.analyzer.turning_bb_color[turn_type]
                cv2.putText(frame, f"{turn_type.title()}: {count}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 20
        
        # Color legend
        self._draw_color_legend(frame, w)
    
    def _draw_color_legend(self, frame, frame_width):
        """Draw color legend for turn types"""
        legend_x = frame_width - 150
        legend_y = 20
        
        cv2.rectangle(frame, (legend_x-10, legend_y-10), (frame_width-10, legend_y+60), (0, 0, 0), -1)
        cv2.putText(frame, "Turn Color:", (legend_x, legend_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Single color box for all turns
        cv2.rectangle(frame, (legend_x, legend_y+20), (legend_x+15, legend_y+30), 
                     self.analyzer.SINGLE_TURN_COLOR, -1)
        cv2.putText(frame, "All Turns", (legend_x+20, legend_y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def process_video(video_path, output_path=None):    
    """
    Main function to process video and analyze traffic.
    
    Process:
    1. Initialize
    2. Validate input and setup output
    3. Process frames sequentially
    4. Generate analytics and reports
    File existence validation
    
    """
    # Validate input
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return None, None
    
    # Initialize components
    analyzer = TrafficAnalyzer()
    processor = VideoProcessor(analyzer)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None
    
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {w}x{h} at {fps} FPS, {total_frames} frames")
    
    # Setup zones and output
    zones = analyzer.define_intersection_zones(w, h)
    turn_map = analyzer.get_zone_turn_mapping()
    
    if output_path is None:
        output_path = "traffic_analysis_output.mp4"
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print("Error: Could not open output video writer")
        cap.release()
        return None, None
    
    # Process frames
    frame_count = 0
    print("Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Show progress
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # Process frame
        frame = process_frame(frame, analyzer, processor, zones, turn_map)
        out.write(frame)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Generate analytics
    analytics = generate_analytics(analyzer, frame_count, output_path)
    
    return output_path, analytics


def process_frame(frame, analyzer, processor, zones, turn_map):
    """
    Process a single frame for vehicle detection and tracking.
    
    Frame Processing Pipeline:
    1. YOLO object detection
    2. Filter for vehicle classes
    3. Update DeepSORT tracker
    4. Analyze trajectories for turn classification
    5. Render visualization overlays
    """
    # Detect vehicles
    results = analyzer.model(frame, imgsz=1280, conf=0.35)[0]
    
    if results.boxes is None or len(results.boxes) == 0:
        processor.draw_polygon_zones(frame, zones)
        processor.draw_debug_info(frame, 0, 0, analyzer.turn_counts, analyzer.total_cars)
        return frame
    
    # Process detections
    cls, xyxy, conf, xywh = processor.get_box_details(results.boxes)
    detections = []
    total_detected = 0
    
    for i, (box, c) in enumerate(zip(xyxy, cls)):
        class_id = int(c)
        if class_id in analyzer.CAR_CLASSES:
            total_detected += 1
            bbox = box.cpu().numpy()
            x1, y1, x2, y2 = bbox
            conf_val = float(conf[i])
            
            if conf_val > 0.45:
                class_name = {2: "car", 5: "bus", 7: "truck"}[class_id]
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf_val, class_name))
    
    # Update tracker
    outputs = analyzer.tracker.update_tracks(detections, frame=frame)
    current_tracks = set()
    valid_tracked = 0
    
    # Process tracked vehicles
    for track in outputs:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        valid_tracked += 1
        track_id = track.track_id
        current_tracks.add(track_id)
        
        # Get bounding box and center
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # Initialize new tracks
        if track_id not in analyzer.trajectories:
            analyzer.total_cars += 1
            analyzer.trajectories[track_id] = []
        
        # Update trajectory
        analyzer.trajectories[track_id].append((cx, cy))
        
        # Get current turn type
        current_turn_type = analyzer.car_turns.get(track_id, None)
        
        # Draw bounding box
        processor.draw_enhanced_bbox(frame, x1, y1, x2, y2, track_id, current_turn_type)
        
        # Draw trajectory
        trajectory_color = analyzer.turning_bb_color.get(current_turn_type, analyzer.SINGLE_TURN_COLOR)
        if current_turn_type is None:
            trajectory_color = (255, 255, 255)
        
        for i, pt in enumerate(analyzer.trajectories[track_id]):
            alpha = max(0.3, (i / len(analyzer.trajectories[track_id])))
            point_color = tuple(int(c * alpha) for c in trajectory_color)
            cv2.circle(frame, pt, 2, point_color, -1)
        
        # Analyze trajectory
        if len(analyzer.trajectories[track_id]) >= 20 and track_id not in analyzer.car_turns:
            analysis = analyzer.analyze_trajectory(analyzer.trajectories[track_id], zones)
            if analysis["valid"] and analysis["turn_type"]:
                turn_type = analysis["turn_type"]
                analyzer.car_turns[track_id] = turn_type
                analyzer.turn_counts[turn_type] += 1
    
    # Handle lost tracks
    lost_tracks = analyzer.previous_tracks - current_tracks
    for track_id in lost_tracks:
        if track_id in analyzer.trajectories and track_id not in analyzer.car_turns:
            if len(analyzer.trajectories[track_id]) >= 15:
                analysis = analyzer.analyze_trajectory(analyzer.trajectories[track_id], zones)
                if analysis["valid"] and analysis["turn_type"]:
                    analyzer.car_turns[track_id] = analysis["turn_type"]
                    analyzer.turn_counts[analysis["turn_type"]] += 1
    
    analyzer.previous_tracks = current_tracks.copy()
    
    # Draw overlays
    processor.draw_polygon_zones(frame, zones)
    processor.draw_debug_info(frame, total_detected, valid_tracked, analyzer.turn_counts, analyzer.total_cars)
    
    return frame


def generate_analytics(analyzer, frame_count, output_path):
    """Generate analytics report"""
    analytics = {
        "total_cars": analyzer.total_cars,
        "turn_counts": analyzer.turn_counts,
        "car_turns": analyzer.car_turns,
    }
    
    # Save analytics file
    with open("turn_analytics.json", "w") as f:
        json.dump(analytics, f, indent=4)
    
    # Print results
    print("Processing complete!")
    print(f"Successfully classified: {len(analyzer.car_turns)} vehicles")
    print(f"Turn counts: {analyzer.turn_counts}")
    print(f"Output video saved as: {output_path}")
    print("Analytics saved as: turn_analytics.json")
    
    # Verify output file
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Output video file size: {file_size / (1024*1024):.2f} MB")
    else:
        print("Warning: Output video file was not created!")
    
    return analytics


if __name__ == "__main__":
    # Configuration
    input_video = "/content/traffic_chaos 2.mp4" 
    output_video = "/content/traffic_analysis_output.mp4"
    
    # Process the video
    result_path, analytics = process_video(input_video, output_video)
    
    if result_path:
        print(f"Success! Processed video saved at: {result_path}")
    else:
        print("Failed to process video. Please check the input path and try again.")



