import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Initialize MediaPipe components
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class AdvancedEmotionFeatureExtractor:
    """Advanced feature extraction for emotion detection from facial landmarks"""
    
    def __init__(self):
        # Facial landmarks indices for different regions (MediaPipe Face Mesh has 468 points)
        self.landmark_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
            'left_eyebrow': [70, 63, 105, 66, 107],
            'right_eyebrow': [300, 293, 334, 296, 336],
            'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
            'nose': [1, 2, 98, 327],
            'jawline': [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
        }
        
        # Emotion categories
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Feature buffers for temporal analysis
        self.feature_history = deque(maxlen=30)  # Store last 30 frames
        self.emotion_history = deque(maxlen=30)
        
        # Initialize feature scaler
        self.scaler = StandardScaler()
        
    def extract_geometric_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Extract geometric features from facial landmarks"""
        features = {}
        
        # 1. Eye features
        left_eye_pts = landmarks[self.landmark_indices['left_eye']]
        right_eye_pts = landmarks[self.landmark_indices['right_eye']]
        
        # Eye aspect ratio (EAR) - measure of eye openness
        features['left_ear'] = self._eye_aspect_ratio(left_eye_pts)
        features['right_ear'] = self._eye_aspect_ratio(right_eye_pts)
        features['ear_avg'] = (features['left_ear'] + features['right_ear']) / 2
        features['ear_diff'] = abs(features['left_ear'] - features['right_ear'])
        
        # 2. Eyebrow features
        left_brow = landmarks[self.landmark_indices['left_eyebrow']]
        right_brow = landmarks[self.landmark_indices['right_eyebrow']]
        features['brow_raise'] = self._eyebrow_raise_degree(left_brow, right_brow)
        
        # 3. Mouth features
        mouth_pts = landmarks[self.landmark_indices['mouth']]
        features['mar'] = self._mouth_aspect_ratio(mouth_pts)
        features['mouth_width'] = self._mouth_width(mouth_pts)
        features['mouth_openness'] = self._mouth_openness(mouth_pts)
        
        # 4. Nose features
        nose_pts = landmarks[self.landmark_indices['nose']]
        features['nose_wrinkle'] = self._nose_wrinkle_degree(nose_pts)
        
        # 5. Asymmetry features (important for detecting disgust, contempt)
        features['face_asymmetry'] = self._calculate_face_asymmetry(landmarks)
        
        # 6. Dynamic features (if history available)
        if len(self.feature_history) > 0:
            prev_features = self.feature_history[-1]
            features['ear_change_rate'] = abs(features['ear_avg'] - prev_features['ear_avg'])
            features['mar_change_rate'] = abs(features['mar'] - prev_features['mar'])
        
        # Store current features in history
        self.feature_history.append(features.copy())
        
        return features
    
    def _eye_aspect_ratio(self, eye_pts: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio"""
        # Vertical distances
        v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
        v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_pts[0] - eye_pts[3])
        
        return (v1 + v2) / (2.0 * h)
    
    def _mouth_aspect_ratio(self, mouth_pts: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio"""
        # Vertical distances
        v1 = np.linalg.norm(mouth_pts[13] - mouth_pts[19])
        v2 = np.linalg.norm(mouth_pts[14] - mouth_pts[18])
        v3 = np.linalg.norm(mouth_pts[15] - mouth_pts[17])
        
        # Horizontal distance
        h = np.linalg.norm(mouth_pts[12] - mouth_pts[16])
        
        return (v1 + v2 + v3) / (3.0 * h)
    
    def _eyebrow_raise_degree(self, left_brow: np.ndarray, right_brow: np.ndarray) -> float:
        """Calculate eyebrow raise degree"""
        left_raise = np.mean(left_brow[:, 1])
        right_raise = np.mean(right_brow[:, 1])
        return (left_raise + right_raise) / 2
    
    def _mouth_width(self, mouth_pts: np.ndarray) -> float:
        """Calculate mouth width"""
        return np.linalg.norm(mouth_pts[12] - mouth_pts[16])
    
    def _mouth_openness(self, mouth_pts: np.ndarray) -> float:
        """Calculate mouth openness"""
        upper_lip = mouth_pts[13]
        lower_lip = mouth_pts[14]
        return np.linalg.norm(upper_lip - lower_lip)
    
    def _nose_wrinkle_degree(self, nose_pts: np.ndarray) -> float:
        """Calculate nose wrinkle degree"""
        return np.std(nose_pts[:, 1])  # Vertical variation
    
    def _calculate_face_asymmetry(self, landmarks: np.ndarray) -> float:
        """Calculate facial asymmetry score"""
        # Compare left and right side landmarks
        left_side = landmarks[:234]  # First half
        right_side = landmarks[234:]  # Second half
        
        # Mirror right side for comparison
        right_side_mirrored = right_side.copy()
        right_side_mirrored[:, 0] = 1.0 - right_side_mirrored[:, 0]  # Mirror horizontally
        
        # Calculate asymmetry score
        asymmetry = np.mean(np.abs(left_side - right_side_mirrored))
        return asymmetry
    
    def extract_texture_features(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Extract texture-based features from face ROI"""
        features = {}
        
        if face_roi is None or face_roi.size == 0:
            return features
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. LBP-like texture features
        features['texture_std'] = np.std(gray)
        features['texture_mean'] = np.mean(gray)
        
        # 2. Edge density
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 3. Local binary patterns (simplified)
        features['local_contrast'] = np.std(cv2.boxFilter(gray, -1, (3, 3)))
        
        return features
    
    def extract_temporal_features(self) -> Dict[str, float]:
        """Extract temporal features from feature history"""
        features = {}
        
        if len(self.feature_history) < 2:
            return features
        
        # Convert history to arrays
        ear_history = [f['ear_avg'] for f in self.feature_history]
        mar_history = [f['mar'] for f in self.feature_history]
        
        # Calculate temporal variations
        features['ear_variation'] = np.std(ear_history)
        features['mar_variation'] = np.std(mar_history)
        
        # Calculate trends
        if len(ear_history) >= 5:
            features['ear_trend'] = np.polyfit(range(len(ear_history)), ear_history, 1)[0]
            features['mar_trend'] = np.polyfit(range(len(mar_history)), mar_history, 1)[0]
        
        return features

class EmotionClassifier:
    """Advanced emotion classifier using feature fusion"""
    
    def __init__(self):
        # Emotion weights based on feature importance
        self.emotion_weights = {
            'angry': {'brow_raise': 0.3, 'mar': 0.2, 'face_asymmetry': 0.2, 'texture_std': 0.3},
            'happy': {'mar': 0.4, 'mouth_openness': 0.3, 'ear_avg': 0.3},
            'sad': {'brow_raise': 0.4, 'ear_avg': 0.3, 'mar': 0.3},
            'surprise': {'ear_avg': 0.4, 'mar': 0.4, 'brow_raise': 0.2},
            'neutral': {'ear_variation': 0.5, 'mar_variation': 0.5}
        }
        
    def classify(self, features: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """Classify emotion based on extracted features"""
        
        emotion_scores = {}
        
        # Calculate scores for each emotion
        for emotion, weights in self.emotion_weights.items():
            score = 0
            total_weight = 0
            
            for feature, weight in weights.items():
                if feature in features:
                    # Normalize feature value (assuming typical ranges)
                    norm_value = self._normalize_feature(feature, features[feature])
                    score += norm_value * weight
                    total_weight += weight
            
            if total_weight > 0:
                emotion_scores[emotion] = score / total_weight
            else:
                emotion_scores[emotion] = 0
        
        # Add dynamic adjustment based on temporal features
        if 'ear_trend' in features:
            if features['ear_trend'] > 0.1:  # Eyes opening
                emotion_scores['surprise'] *= 1.5
            elif features['ear_trend'] < -0.1:  # Eyes closing
                emotion_scores['sad'] *= 1.3
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # Apply confidence threshold
        confidence = emotion_scores[dominant_emotion]
        if confidence < 0.3:
            dominant_emotion = 'neutral'
        
        return dominant_emotion, emotion_scores
    
    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize feature values to [0, 1] range"""
        normalization_ranges = {
            'ear_avg': (0.15, 0.4),
            'mar': (0.05, 0.8),
            'brow_raise': (0.3, 0.7),
            'mouth_openness': (0.01, 0.2),
            'face_asymmetry': (0.0, 0.1),
            'texture_std': (10, 50),
            'ear_variation': (0.0, 0.1),
            'mar_variation': (0.0, 0.2)
        }
        
        if feature_name in normalization_ranges:
            min_val, max_val = normalization_ranges[feature_name]
            return max(0, min(1, (value - min_val) / (max_val - min_val)))
        
        return max(0, min(1, value))

class RealTimeEmotionDetector:
    """Main class for real-time emotion detection"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        
        # Initialize components
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.feature_extractor = AdvancedEmotionFeatureExtractor()
        self.emotion_classifier = EmotionClassifier()
        
        # Visualization parameters
        self.colors = {
            'angry': (0, 0, 255),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'surprise': (255, 255, 0),
            'neutral': (200, 200, 200),
            'disgust': (0, 255, 255),
            'fear': (255, 0, 255)
        }
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        
        # Emotion statistics
        self.emotion_counts = {emotion: 0 for emotion in self.feature_extractor.emotions}
        self.session_start_time = time.time()
        
    def start(self):
        """Start the real-time emotion detection"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.running = True
        
        # Create display window
        cv2.namedWindow('Advanced Emotion Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Advanced Emotion Detection', 1200, 800)
        
        print("Starting real-time emotion detection...")
        print("Press 'q' to quit")
        print("Press 's' to save statistics")
        print("Press 'r' to reset statistics")
        
        while self.running:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            emotion = "neutral"
            emotion_scores = {}
            landmarks = None
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract landmarks
                    landmarks = np.array([[lm.x * frame_width, lm.y * frame_height, lm.z * frame_width] 
                                          for lm in face_landmarks.landmark])
                    
                    # Extract face ROI for texture features
                    x_min, y_min = landmarks[:, :2].min(axis=0).astype(int)
                    x_max, y_max = landmarks[:, :2].max(axis=0).astype(int)
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame_width, x_max + padding)
                    y_max = min(frame_height, y_max + padding)
                    
                    face_roi = frame[y_min:y_max, x_min:x_max]
                    
                    # Extract features
                    geometric_features = self.feature_extractor.extract_geometric_features(landmarks)
                    texture_features = self.feature_extractor.extract_texture_features(face_roi)
                    temporal_features = self.feature_extractor.extract_temporal_features()
                    
                    # Combine all features
                    all_features = {**geometric_features, **texture_features, **temporal_features}
                    
                    # Classify emotion
                    emotion, emotion_scores = self.emotion_classifier.classify(all_features)
                    
                    # Update statistics
                    self.emotion_counts[emotion] += 1
                    
                    # Draw landmarks and annotations
                    self._draw_face_analysis(frame, landmarks, emotion, emotion_scores, face_roi)
                    
                    # Draw feature visualization
                    self._draw_feature_visualization(frame, geometric_features, (frame_width - 300, 50))
            
            # Calculate FPS
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            fps = 1.0 / processing_time if processing_time > 0 else 0
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history) if self.fps_history else 0
            
            # Display FPS and statistics
            self._display_statistics(frame, avg_fps, emotion_scores)
            
            # Show frame
            cv2.imshow('Advanced Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self.save_statistics()
            elif key == ord('r'):
                self.reset_statistics()
            elif key == ord('v'):
                self.toggle_visualization()
    
    def _draw_face_analysis(self, frame: np.ndarray, landmarks: np.ndarray, 
                          emotion: str, emotion_scores: Dict[str, float], face_roi: np.ndarray):
        """Draw face analysis results on frame"""
        
        # Draw face mesh
        # Uncomment to see full face mesh
        # mp_drawing.draw_landmarks(
        #     image=frame,
        #     landmark_list=mp.solutions.face_mesh.FaceMesh,
        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        # )
        
        # Draw important landmarks with colors
        for region_name, indices in self.feature_extractor.landmark_indices.items():
            color = (0, 255, 0) if region_name in ['left_eye', 'right_eye', 'mouth'] else (255, 0, 0)
            for idx in indices:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
                    cv2.circle(frame, (x, y), 2, color, -1)
        
        # Draw emotion text with background
        emotion_color = self.colors.get(emotion, (255, 255, 255))
        text = f"Emotion: {emotion.upper()}"
        font_scale = 1.2
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Background rectangle
        cv2.rectangle(frame, (10, 10), (10 + text_width + 20, 10 + text_height + 20), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, emotion_color, thickness)
        
        # Draw emotion scores
        y_offset = 80
        for emo, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
            color = self.colors.get(emo, (255, 255, 255))
            bar_width = int(score * 200)
            cv2.rectangle(frame, (20, y_offset), (20 + bar_width, y_offset + 20), color, -1)
            cv2.putText(frame, f"{emo}: {score:.2f}", (230, y_offset + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
    
    def _draw_feature_visualization(self, frame: np.ndarray, features: Dict[str, float], position: Tuple[int, int]):
        """Draw feature visualization sidebar"""
        x, y = position
        
        # Draw feature visualization background
        cv2.rectangle(frame, (x, y), (x + 280, y + 200), (40, 40, 40), -1)
        cv2.putText(frame, "FEATURE VALUES", (x + 10, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display key features
        key_features = ['ear_avg', 'mar', 'brow_raise', 'mouth_openness', 'face_asymmetry']
        y_offset = y + 40
        
        for feat in key_features:
            if feat in features:
                value = features[feat]
                # Normalize for display
                display_value = min(100, max(0, int(value * 100)))
                
                # Draw bar
                bar_width = display_value * 2
                cv2.rectangle(frame, (x + 10, y_offset), (x + 10 + bar_width, y_offset + 15), 
                            (0, 200, 200), -1)
                
                # Draw text
                cv2.putText(frame, f"{feat}: {value:.3f}", (x + 10, y_offset + 12), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                y_offset += 25
    
    def _display_statistics(self, frame: np.ndarray, fps: float, emotion_scores: Dict[str, float]):
        """Display statistics on frame"""
        # FPS display
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Session duration
        duration = time.time() - self.session_start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", 
                   (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Emotion distribution
        total_detections = sum(self.emotion_counts.values())
        if total_detections > 0:
            y_offset = 100
            cv2.putText(frame, "Emotion Distribution:", (frame.shape[1] - 200, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for emotion, count in sorted(self.emotion_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / total_detections) * 100
                    y_offset += 20
                    color = self.colors.get(emotion, (255, 255, 255))
                    cv2.putText(frame, f"{emotion}: {percentage:.1f}%", 
                              (frame.shape[1] - 200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def save_statistics(self):
        """Save session statistics to file"""
        filename = f"emotion_stats_{time.strftime('%Y%m%d_%H%M%S')}.json"
        stats = {
            'session_duration': time.time() - self.session_start_time,
            'emotion_counts': self.emotion_counts,
            'average_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to {filename}")
        
        # Also create visualization
        self.create_statistics_plot(filename.replace('.json', '.png'))
    
    def create_statistics_plot(self, filename: str):
        """Create and save statistics visualization plot"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Emotion distribution pie chart
        emotions = list(self.emotion_counts.keys())
        counts = list(self.emotion_counts.values())
        
        # Filter out zero counts
        data = [(e, c) for e, c in zip(emotions, counts) if c > 0]
        if data:
            emotions_filtered, counts_filtered = zip(*data)
            
            colors_plot = [self.colors[e] for e in emotions_filtered]
            colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors_plot]
            
            axes[0].pie(counts_filtered, labels=emotions_filtered, colors=colors_normalized, 
                       autopct='%1.1f%%', startangle=90)
            axes[0].set_title('Emotion Distribution')
        
        # FPS history plot
        if self.fps_history:
            axes[1].plot(list(self.fps_history), marker='o', linestyle='-', color='b')
            axes[1].axhline(y=np.mean(self.fps_history), color='r', linestyle='--', 
                          label=f'Avg: {np.mean(self.fps_history):.1f} FPS')
            axes[1].set_xlabel('Frame')
            axes[1].set_ylabel('FPS')
            axes[1].set_title('Processing Performance')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {filename}")
    
    def reset_statistics(self):
        """Reset session statistics"""
        self.emotion_counts = {emotion: 0 for emotion in self.feature_extractor.emotions}
        self.session_start_time = time.time()
        self.fps_history.clear()
        self.processing_times.clear()
        print("Statistics reset")
    
    def toggle_visualization(self):
        """Toggle visualization modes"""
        # This can be extended to toggle different visualization modes
        print("Visualization toggle - Implement custom visualization modes here")
    
    def stop(self):
        """Stop the detector and release resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Emotion detector stopped")

class BatchEmotionAnalyzer:
    """Analyze emotions from video files or image sequences"""
    
    @staticmethod
    def analyze_video(video_path: str, output_path: str = None):
        """Analyze emotions from a video file"""
        detector = RealTimeEmotionDetector()
        detector.cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(detector.cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(detector.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(detector.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        detector.running = True
        
        while detector.running:
            ret, frame = detector.cap.read()
            if not ret:
                break
            
            # Process frame (simplified - would need to integrate main processing logic)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.face_mesh.process(rgb_frame)
            
            if output_path:
                out.write(frame)
        
        if output_path:
            out.release()
        detector.stop()

def main():
    """Main function to run the emotion detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Real-Time Emotion Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--video', type=str, help='Path to video file for analysis')
    parser.add_argument('--output', type=str, help='Output path for analyzed video')
    
    args = parser.parse_args()
    
    if args.video:
        print(f"Analyzing video: {args.video}")
        BatchEmotionAnalyzer.analyze_video(args.video, args.output)
    else:
        detector = RealTimeEmotionDetector(camera_id=args.camera)
        try:
            detector.start()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            detector.stop()
            
            # Save final statistics
            detector.save_statistics()

if __name__ == "__main__":
    main()
