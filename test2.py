import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from PIL import Image, ImageTk
import mediapipe as mp
import pickle
import math
from scipy.spatial.distance import cosine
import urllib.request
import zipfile
import io
import shutil

# Global constants
APP_TITLE = "Face Recognition System"
FACE_DB_PATH = "face_database"
MODEL_PATH = "models"
FACE_DETECTION_THRESHOLD = 0.5
FACE_RECOGNITION_THRESHOLD = 0.2  # Lower value = stricter matching (changed from 0.3)

# OpenCV Haar Cascade as fallback
CASCADE_PATH = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x700")
        self.root.resizable(True, True)

        # Create directories first
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(FACE_DB_PATH, exist_ok=True)

        # Initialize variables
        self.camera_index = 0
        self.available_cameras = self.get_available_cameras()
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.face_database = self.load_face_database()
        self.recognition_mode = tk.StringVar(value="Recognition")
        self.person_name = tk.StringVar()
        self.accuracy_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "total_frames": 0
        }

        # Init status for models
        self.using_insightface = False
        self.using_mediapipe = False

        # Download cascade classifier as fallback
        self.download_cascade_classifier()

        # Setup UI
        self.setup_ui()

        # Initialize face detection and recognition models (in a separate thread)
        self.status_var = tk.StringVar(value="Initializing models...")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var,
                                      foreground="blue", font=("Arial", 10, "italic"))
        self.status_label.pack(anchor=tk.W, pady=(5, 10))

        # Start model initialization in background
        threading.Thread(target=self.setup_models, daemon=True).start()

    def download_cascade_classifier(self):
        """Download the Haar Cascade classifier file if needed"""
        if not os.path.exists(CASCADE_PATH):
            try:
                response = urllib.request.urlopen(CASCADE_URL)
                data = response.read()
                with open(CASCADE_PATH, 'wb') as f:
                    f.write(data)
                print(f"Downloaded Haar Cascade classifier to {CASCADE_PATH}")
            except Exception as e:
                print(f"Error downloading Haar Cascade classifier: {e}")

    def setup_models(self):
        """Initialize face detection and recognition models"""
        self.status_var.set("Initializing MediaPipe Face Mesh...")

        try:
            # Initialize MediaPipe Face Mesh for additional facial landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.using_mediapipe = True
        except Exception as e:
            self.status_var.set(f"MediaPipe initialization failed: {str(e)}")
            self.using_mediapipe = False

        # Initialize fallback face detector from OpenCV
        self.status_var.set("Loading Cascade Classifier (fallback)...")
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        # Try to initialize InsightFace
        try:
            self.status_var.set("Trying to import InsightFace...")
            import insightface
            from insightface.app import FaceAnalysis

            self.status_var.set("Initializing SCRFD face detector via InsightFace...")

            # Initialize SCRFD face detector via InsightFace
            self.face_analyzer = FaceAnalysis(
                name="buffalo_l",  # Using the large model for better accuracy
                root=MODEL_PATH,
                providers=['CPUExecutionProvider']  # Use CPU, change to CUDAExecutionProvider for GPU
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

            # Check if MobileFaceNet model exists
            model_path = os.path.join(MODEL_PATH, 'mobilefacenet.onnx')
            if not os.path.exists(model_path):
                self.status_var.set("Downloading MobileFaceNet model...")
                try:
                    # Let's try to download it explicitly
                    # Note: this is a placeholder URL, replace with actual model URL
                    model_url = "https://github.com/deepinsight/insightface/raw/master/python-package/insightface/model_zoo/models/mobilefacenet.onnx"
                    urllib.request.urlretrieve(model_url, model_path)
                except Exception as e:
                    self.status_var.set(f"Model download failed: {str(e)}. Using alternative approach.")
                    # If direct download fails, we'll use face_analyzer for recognition too
                    pass

            # If model exists, try to load it
            if os.path.exists(model_path):
                self.status_var.set("Loading MobileFaceNet model...")
                try:
                    self.face_recognizer = insightface.model_zoo.get_model(
                        model_path,
                        providers=['CPUExecutionProvider']
                    )
                    self.status_var.set("InsightFace models loaded successfully!")
                    self.using_insightface = True
                except Exception as e:
                    self.status_var.set(f"Error loading MobileFaceNet: {str(e)}. Using face_analyzer for recognition.")
                    # We'll use face_analyzer for recognition too
                    self.using_insightface = True
            else:
                # We'll use face_analyzer for both detection and recognition
                self.status_var.set("Using face_analyzer for both detection and recognition.")
                self.using_insightface = True

        except ImportError:
            self.status_var.set("InsightFace not available. Using OpenCV and MediaPipe only.")
            self.using_insightface = False
        except Exception as e:
            self.status_var.set(f"InsightFace initialization error: {str(e)}. Using fallback methods.")
            self.using_insightface = False

        # Final status update
        if self.using_insightface and self.using_mediapipe:
            self.status_var.set("All models loaded successfully! Full functionality available.")
        elif self.using_insightface:
            self.status_var.set("InsightFace loaded. MediaPipe unavailable.")
        elif self.using_mediapipe:
            self.status_var.set("Using MediaPipe and OpenCV. InsightFace unavailable.")
        else:
            self.status_var.set("Using OpenCV only. Limited functionality.")

    def get_available_cameras(self):
        """Detect available camera devices"""
        available_cameras = []
        max_test = 10  # Test up to 10 camera indices

        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    camera_name = f"Camera {i}"
                    try:
                        # Try to get camera name/description
                        camera_name = f"Camera {i}: {cap.getBackendName()}"
                    except:
                        pass
                    available_cameras.append((camera_name, i))
                cap.release()

        # If no cameras found, add a placeholder
        if not available_cameras:
            available_cameras.append(("No cameras found", -1))

        return available_cameras

    def load_face_database(self):
        """Load the saved face embeddings database"""
        db_file = os.path.join(FACE_DB_PATH, "face_db.pkl")
        if os.path.exists(db_file):
            try:
                with open(db_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading face database: {e}")
                return {}
        return {}

    def save_face_database(self):
        """Save the face embeddings database"""
        db_file = os.path.join(FACE_DB_PATH, "face_db.pkl")
        try:
            with open(db_file, 'wb') as f:
                pickle.dump(self.face_database, f)
            print("Face database saved successfully")
        except Exception as e:
            print(f"Error saving face database: {e}")

    def setup_ui(self):
        """Setup the Tkinter user interface"""
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.video_frame = ttk.Frame(self.root, padding=10)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.stats_frame = ttk.Frame(self.root, padding=10, width=200)
        self.stats_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Camera controls section
        ttk.Label(self.control_frame, text="Camera Controls", font=("Arial", 12, "bold")).pack(anchor=tk.W,
                                                                                               pady=(0, 10))

        # Camera selection dropdown
        ttk.Label(self.control_frame, text="Select Camera:").pack(anchor=tk.W)
        self.camera_combo = ttk.Combobox(self.control_frame, state="readonly", width=30)
        self.camera_combo['values'] = [cam[0] for cam in self.available_cameras]
        self.camera_combo.current(0)
        self.camera_combo.pack(anchor=tk.W, pady=(0, 10))
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change)

        # Start/Stop button
        self.start_stop_btn = ttk.Button(self.control_frame, text="Start Camera", command=self.toggle_camera)
        self.start_stop_btn.pack(anchor=tk.W, pady=(0, 20))

        # Mode selection
        ttk.Label(self.control_frame, text="Operation Mode:", font=("Arial", 12, "bold")).pack(anchor=tk.W,
                                                                                               pady=(0, 10))
        ttk.Radiobutton(self.control_frame, text="Recognition Mode", variable=self.recognition_mode,
                        value="Recognition").pack(anchor=tk.W)
        ttk.Radiobutton(self.control_frame, text="Training Mode", variable=self.recognition_mode,
                        value="Training").pack(anchor=tk.W, pady=(0, 10))

        # Person name entry (for training)
        ttk.Label(self.control_frame, text="Person Name:").pack(anchor=tk.W)
        ttk.Entry(self.control_frame, textvariable=self.person_name).pack(anchor=tk.W, fill=tk.X, pady=(0, 10))

        # Training controls
        ttk.Label(self.control_frame, text="Training Controls", font=("Arial", 12, "bold")).pack(anchor=tk.W,
                                                                                                 pady=(10, 10))

        # Add face buttons
        ttk.Button(self.control_frame, text="Add Face from Camera", command=self.add_face_from_camera).pack(anchor=tk.W,
                                                                                                            pady=(0, 5))
        ttk.Button(self.control_frame, text="Add Multiple Angles (Recommended)", 
                   command=lambda: self.add_face_sequence(self.person_name.get())).pack(
                   anchor=tk.W, pady=(0, 5))
        ttk.Button(self.control_frame, text="Add Face from Image", command=self.add_face_from_image).pack(anchor=tk.W,
                                                                                                          pady=(0, 5))
        ttk.Button(self.control_frame, text="Add Face from Video", command=self.add_face_from_video).pack(anchor=tk.W,
                                                                                                          pady=(0, 20))

        # Database management
        ttk.Label(self.control_frame, text="Database Controls", font=("Arial", 12, "bold")).pack(anchor=tk.W,
                                                                                                 pady=(10, 10))
        ttk.Button(self.control_frame, text="View Face Database", command=self.view_face_database).pack(anchor=tk.W,
                                                                                                        pady=(0, 5))
        ttk.Button(self.control_frame, text="Clear Face Database", command=self.clear_face_database).pack(anchor=tk.W,
                                                                                                          pady=(0, 20))

        # Accuracy testing controls
        ttk.Label(self.control_frame, text="Accuracy Testing", font=("Arial", 12, "bold")).pack(anchor=tk.W,
                                                                                                pady=(10, 10))
        ttk.Button(self.control_frame, text="Start Accuracy Test", command=self.start_accuracy_test).pack(anchor=tk.W,
                                                                                                          pady=(0, 5))
        ttk.Button(self.control_frame, text="Reset Metrics", command=self.reset_accuracy_metrics).pack(anchor=tk.W,
                                                                                                       pady=(0, 5))

        # Video display area
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Stats display area
        ttk.Label(self.stats_frame, text="Statistics", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))

        self.stats_display = ttk.Label(self.stats_frame, text="No data available", justify=tk.LEFT)
        self.stats_display.pack(anchor=tk.W, fill=tk.X)

        # Start stats updating
        self.update_stats()

    def on_camera_change(self, event):
        """Handle camera selection change"""
        if self.is_running:
            self.toggle_camera()  # Stop the current camera

        # Get the selected camera index
        selected_idx = self.camera_combo.current()
        if 0 <= selected_idx < len(self.available_cameras):
            self.camera_index = self.available_cameras[selected_idx][1]

    def toggle_camera(self):
        """Start or stop the camera"""
        if self.is_running:
            # Stop the camera
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.start_stop_btn.config(text="Start Camera")
        else:
            # Start the camera
            self.is_running = True
            self.start_stop_btn.config(text="Stop Camera")

            # Start video capture in a separate thread
            threading.Thread(target=self.video_loop, daemon=True).start()

    def video_loop(self):
        """Main video processing loop"""
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open camera {self.camera_index}")
            self.is_running = False
            self.start_stop_btn.config(text="Start Camera")
            return

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Create a copy of the frame for processing
            self.current_frame = frame.copy()

            # Process the frame based on current mode
            if self.recognition_mode.get() == "Recognition":
                self.process_recognition(frame)
            else:
                # Just show detected faces in training mode
                self.process_training_preview(frame)

            # Convert to format suitable for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the video label (needs to be in the main thread)
            self.root.after(0, lambda: self.video_label.config(image=imgtk))
            self.video_label.image = imgtk

            # Update metrics if we're in testing mode
            self.accuracy_metrics["total_frames"] += 1

            # Control the frame rate
            time.sleep(0.02)  # ~50 FPS maximum

        if self.cap is not None:
            self.cap.release()

    def detect_faces_fallback(self, frame):
        """Detect faces using OpenCV Cascade Classifier as fallback"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces  # Returns list of (x, y, w, h) tuples

    def get_face_embedding_fallback(self, frame, face_rect):
        """Generate a simple face embedding using pixel values (fallback)"""
        x, y, w, h = face_rect
        # Extract face region and resize to fixed size
        face_region = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_region, (64, 64))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # Flatten and normalize pixels as a very basic "embedding"
        # This is NOT a good embedding, just a fallback
        embedding = face_gray.flatten().astype(np.float32) / 255.0
        return embedding

    def process_recognition(self, frame):
        """Process frame for face recognition"""
        if self.using_insightface:
            # Use InsightFace for detection
            try:
                faces = self.face_analyzer.get(frame)

                for face in faces:
                    box = face.bbox.astype(int)
                    # Extract face coordinates
                    x1, y1, x2, y2 = box

                    # Draw face rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Get face embedding
                    embedding = face.embedding

                    # Match with known faces
                    if embedding is not None and self.face_database:
                        # Find the best match
                        best_match_name, best_match_dist = self.find_face_match(embedding)

                        if best_match_dist < FACE_RECOGNITION_THRESHOLD:
                            # Calculate confidence percentage (inversely related to distance)
                            confidence = (1 - best_match_dist) * 100
                            label = f"{best_match_name} ({confidence:.1f}%)"
                            color = (0, 255, 0)  # Green for recognized

                            # Count as true positive for accuracy metrics
                            self.accuracy_metrics["true_positives"] += 1
                        else:
                            label = "Unknown"
                            color = (0, 0, 255)  # Red for unknown

                            # Count as false negative
                            self.accuracy_metrics["false_negatives"] += 1

                        # Display name and confidence
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                error_msg = f"InsightFace error: {str(e)}"
                cv2.putText(frame, error_msg, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Use fallback if InsightFace fails
                self.process_recognition_fallback(frame)
        else:
            # Use fallback methods
            self.process_recognition_fallback(frame)

        # Draw MediaPipe landmarks if available
        if self.using_mediapipe:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_results = self.face_mesh.process(rgb_frame)

                if mp_results.multi_face_landmarks:
                    mp_drawing = mp.solutions.drawing_utils
                    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255, 0, 255))

                    for face_landmarks in mp_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            self.mp_face_mesh.FACEMESH_CONTOURS,
                            drawing_spec,
                            drawing_spec
                        )
            except Exception as e:
                mp_error = f"MediaPipe error: {str(e)}"
                cv2.putText(frame, mp_error, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def process_recognition_fallback(self, frame):
        """Process frame using fallback detection and recognition"""
        # Detect faces using Haar cascade
        faces = self.detect_faces_fallback(frame)

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get face embedding
            embedding = self.get_face_embedding_fallback(frame, (x, y, w, h))

            # Match with known faces if we have a database
            if self.face_database:
                # Find the best match
                best_match_name, best_match_dist = self.find_face_match(embedding)

                # Note: we use a different threshold for the fallback method
                fallback_threshold = 0.5  # Higher is more permissive

                if best_match_dist < fallback_threshold:
                    # Calculate confidence percentage
                    confidence = (1 - best_match_dist) * 100
                    label = f"{best_match_name} ({confidence:.1f}%)"
                    color = (0, 255, 0)  # Green for recognized

                    # Count as true positive for accuracy metrics
                    self.accuracy_metrics["true_positives"] += 1
                else:
                    label = "Unknown"
                    color = (0, 0, 255)  # Red for unknown

                    # Count as false negative
                    self.accuracy_metrics["false_negatives"] += 1

                # Display name and confidence
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_training_preview(self, frame):
        """Process frame for training mode preview"""
        if self.using_insightface:
            try:
                # Detect faces using InsightFace
                faces = self.face_analyzer.get(frame)

                for face in faces:
                    box = face.bbox.astype(int)
                    # Extract face coordinates
                    x1, y1, x2, y2 = box

                    # Draw face rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange for training mode

                    # Add label indicating training mode
                    cv2.putText(frame, "Training Mode", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            except Exception as e:
                error_msg = f"InsightFace error: {str(e)}"
                cv2.putText(frame, error_msg, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Use fallback if InsightFace fails
                self.process_training_preview_fallback(frame)
        else:
            # Use fallback method
            self.process_training_preview_fallback(frame)

    def process_training_preview_fallback(self, frame):
        """Process training preview with fallback method"""
        # Detect faces using Haar cascade
        faces = self.detect_faces_fallback(frame)

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)  # Orange for training mode

            # Add label indicating training mode
            cv2.putText(frame, "Training Mode", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    def find_face_match(self, embedding):
        """Find the best match for a face embedding in the database with improved accuracy"""
        best_match_name = "Unknown"
        best_match_dist = float('inf')
        second_best_dist = float('inf')
        
        for name, embeddings_list in self.face_database.items():
            # Calculate average distance to all embeddings for this person
            distances = []
            for stored_embedding in embeddings_list:
                # Make sure shapes match for comparison
                if len(embedding) != len(stored_embedding):
                    continue
                    
                # Calculate cosine distance
                dist = cosine(embedding, stored_embedding)
                distances.append(dist)
            
            if distances:
                # Sort distances and take average of top 3 (or fewer if less available)
                distances.sort()
                avg_top_dist = sum(distances[:min(3, len(distances))]) / min(3, len(distances))
                
                if avg_top_dist < best_match_dist:
                    second_best_dist = best_match_dist
                    best_match_dist = avg_top_dist
                    best_match_name = name
        
        # Calculate distinctiveness ratio (how unique this match is)
        distinctiveness = second_best_dist / best_match_dist if best_match_dist > 0 else 1
        
        # If not distinct enough, be more conservative
        if distinctiveness < 1.5:
            # Require a stricter threshold for non-distinct matches
            adjusted_threshold = FACE_RECOGNITION_THRESHOLD * 0.8
            if best_match_dist > adjusted_threshold:
                best_match_name = "Unknown"
                
        return best_match_name, best_match_dist

    def add_face_sequence(self, name, count=5, delay=1):
        """Capture a sequence of face images with different angles/expressions"""
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start the camera first.")
            return
            
        if not name.strip():
            messagebox.showwarning("Warning", "Please enter a person name.")
            return
            
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Face Capture")
        progress_window.geometry("300x150")
        
        instruction_label = ttk.Label(progress_window, 
                                    text="Please move your face slightly between captures\n"
                                         "to get different angles and expressions")
        instruction_label.pack(pady=10)
        
        progress_label = ttk.Label(progress_window, text=f"Capturing: 0/{count}")
        progress_label.pack(pady=5)
        
        progress_bar = ttk.Progressbar(progress_window, length=250, maximum=count)
        progress_bar.pack(pady=10)
        
        # Function to capture faces
        def capture_sequence():
            captured = 0
            embeddings_added = 0
            last_embedding = None
            
            while captured < count and self.is_running:
                if self.current_frame is not None:
                    # Update progress
                    progress_bar['value'] = captured
                    progress_label.config(text=f"Capturing: {captured}/{count}")
                    progress_window.update()
                    
                    # Detect and add face
                    if self.using_insightface:
                        try:
                            faces = self.face_analyzer.get(self.current_frame)
                            
                            if faces:
                                # Get largest face
                                largest_face = max(faces, 
                                                key=lambda face: (face.bbox[2] - face.bbox[0]) * 
                                                                (face.bbox[3] - face.bbox[1]))
                                
                                # Get embedding
                                embedding = largest_face.embedding
                                
                                # Check if this embedding is different enough from the last one
                                # to ensure we're getting diverse angles
                                is_different = True
                                if last_embedding is not None:
                                    similarity = 1 - cosine(embedding, last_embedding)
                                    # If more than 95% similar to last one, it's too similar
                                    is_different = similarity < 0.95
                                    
                                if is_different:
                                    # Add to database
                                    if name not in self.face_database:
                                        self.face_database[name] = []
                                        
                                    self.face_database[name].append(embedding)
                                    last_embedding = embedding
                                    embeddings_added += 1
                                    
                                    # Save image for reference
                                    os.makedirs(os.path.join(FACE_DB_PATH, "images"), exist_ok=True)
                                    timestamp = int(time.time())
                                    img_path = os.path.join(FACE_DB_PATH, "images", 
                                                         f"{name}_{timestamp}_{captured}.jpg")
                                    
                                    # Draw rectangle
                                    img = self.current_frame.copy()
                                    x1, y1, x2, y2 = map(int, largest_face.bbox)
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.imwrite(img_path, img)
                                    
                                    captured += 1
                        except Exception as e:
                            print(f"Error during sequence capture: {e}")
                    else:
                        # Use fallback method 
                        # Detect faces using Haar cascade
                        faces = self.detect_faces_fallback(self.current_frame)
                        
                        if faces:
                            # Get largest face
                            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                            
                            # Get face embedding
                            embedding = self.get_face_embedding_fallback(self.current_frame, largest_face)
                            
                            # Check if this embedding is different enough
                            is_different = True
                            if last_embedding is not None and len(embedding) == len(last_embedding):
                                similarity = 1 - cosine(embedding, last_embedding)
                                is_different = similarity < 0.95
                                
                            if is_different:
                                # Add to database
                                if name not in self.face_database:
                                    self.face_database[name] = []
                                    
                                self.face_database[name].append(embedding)
                                last_embedding = embedding
                                embeddings_added += 1
                                
                                # Save image
                                os.makedirs(os.path.join(FACE_DB_PATH, "images"), exist_ok=True)
                                timestamp = int(time.time())
                                img_path = os.path.join(FACE_DB_PATH, "images", 
                                                      f"{name}_{timestamp}_{captured}.jpg")
                                
                                # Draw rectangle
                                img = self.current_frame.copy()
                                x, y, w, h = largest_face
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.imwrite(img_path, img)
                                
                                captured += 1
                        
                # Wait before next capture to allow pose change
                time.sleep(delay)
                
            # Save database
            self.save_face_database()
            
            # Show completion message
            self.root.after(0, lambda: messagebox.showinfo("Capture Complete", 
                             f"Added {embeddings_added} face embeddings for {name}"))
            self.root.after(0, progress_window.destroy)
            
        # Start capture in a separate thread
        threading.Thread(target=capture_sequence, daemon=True).start()

    def add_face_from_camera(self):
        """Add a face from the current camera frame"""
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start the camera first.")
            return

        if not self.person_name.get().strip():
            messagebox.showwarning("Warning", "Please enter a person name.")
            return

        name = self.person_name.get().strip()

        if self.current_frame is not None:
            if self.using_insightface:
                # Detect faces using InsightFace
                try:
                    faces = self.face_analyzer.get(self.current_frame)

                    if not faces:
                        messagebox.showwarning("Warning", "No face detected in the current frame.")
                        return

                    # Use the largest face if multiple are detected
                    largest_face = max(faces,
                                       key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))

                    # Get face embedding
                    embedding = largest_face.embedding

                    # Add to database
                    if name not in self.face_database:
                        self.face_database[name] = []

                    self.face_database[name].append(embedding)

                    # Save database
                    self.save_face_database()

                    messagebox.showinfo("Success", f"Face for {name} added successfully.")

                    # Draw rectangle around the detected face and save for confirmation
                    x1, y1, x2, y2 = map(int, largest_face.bbox)
                    cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Save the annotated image
                    os.makedirs(os.path.join(FACE_DB_PATH, "images"), exist_ok=True)
                    timestamp = int(time.time())
                    img_path = os.path.join(FACE_DB_PATH, "images", f"{name}_{timestamp}.jpg")
                    cv2.imwrite(img_path, self.current_frame)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to add face: {str(e)}")
                    # Use fallback method
                    self.add_face_from_camera_fallback()
            else:
                # Use fallback method
                self.add_face_from_camera_fallback()

    def add_face_from_camera_fallback(self):
        """Add a face from the current camera frame using fallback method"""
        if not self.person_name.get().strip():
            messagebox.showwarning("Warning", "Please enter a person name.")
            return

        name = self.person_name.get().strip()

        # Detect faces using Haar cascade
        faces = self.detect_faces_fallback(self.current_frame)

        if len(faces) == 0:
            messagebox.showwarning("Warning", "No face detected in the current frame.")
            return

        # Use the largest face if multiple are detected
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])

        # Get face embedding
        embedding = self.get_face_embedding_fallback(self.current_frame, largest_face)

        # Add to database
        if name not in self.face_database:
            self.face_database[name] = []

        self.face_database[name].append(embedding)

        # Save database
        self.save_face_database()

        messagebox.showinfo("Success", f"Face for {name} added successfully (using fallback method).")

        # Draw rectangle around the detected face and save for confirmation
        x, y, w, h = largest_face
        cv2.rectangle(self.current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the annotated image
        os.makedirs(os.path.join(FACE_DB_PATH, "images"), exist_ok=True)
        timestamp = int(time.time())
        img_path = os.path.join(FACE_DB_PATH, "images", f"{name}_{timestamp}.jpg")
        cv2.imwrite(img_path, self.current_frame)

    def add_face_from_image(self):
        """Add a face from an image file"""
        if not self.person_name.get().strip():
            messagebox.showwarning("Warning", "Please enter a person name.")
            return

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            return

        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Could not read the image file.")
            return

        name = self.person_name.get().strip()
        faces_added = 0

        if self.using_insightface:
            try:
                # Detect faces using InsightFace
                faces = self.face_analyzer.get(image)

                if not faces:
                    messagebox.showwarning("Warning", "No face detected in the image.")
                    return

                # For each face, add it to the database
                for face in faces:
                    # Get face embedding
                    embedding = face.embedding

                    # Add to database
                    if name not in self.face_database:
                        self.face_database[name] = []

                    self.face_database[name].append(embedding)
                    faces_added += 1

                    # Draw rectangle around the detected face
                    x1, y1, x2, y2 = map(int, face.bbox)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception as e:
                messagebox.showerror("Error", f"InsightFace error: {str(e)}. Using fallback method.")
                # Use fallback method
                faces_added = self.add_face_from_image_fallback(image, name)
        else:
            # Use fallback method
            faces_added = self.add_face_from_image_fallback(image, name)

        if faces_added > 0:
            # Save database
            self.save_face_database()

            # Save the annotated image
            os.makedirs(os.path.join(FACE_DB_PATH, "images"), exist_ok=True)
            timestamp = int(time.time())
            img_path = os.path.join(FACE_DB_PATH, "images", f"{name}_{timestamp}.jpg")
            cv2.imwrite(img_path, image)

            messagebox.showinfo("Success", f"{faces_added} face(s) for {name} added successfully.")

    def add_face_from_image_fallback(self, image, name):
        """Add faces from an image using fallback method"""
        # Detect faces using Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            messagebox.showwarning("Warning", "No face detected in the image.")
            return 0

        faces_added = 0

        # For each face, add it to the database
        for (x, y, w, h) in faces:
            # Get face embedding
            embedding = self.get_face_embedding_fallback(image, (x, y, w, h))

            # Add to database
            if name not in self.face_database:
                self.face_database[name] = []

            self.face_database[name].append(embedding)
            faces_added += 1

            # Draw rectangle around the detected face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return faces_added

    def add_face_from_video(self):
        """Add faces from a video file"""
        if not self.person_name.get().strip():
            messagebox.showwarning("Warning", "Please enter a person name.")
            return

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )

        if not file_path:
            return

        # Create a progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Video")
        progress_window.geometry("300x100")

        ttk.Label(progress_window, text="Processing video frames...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, length=250, mode="indeterminate")
        progress_bar.pack(pady=10)
        progress_bar.start()

        # Function to process video in the background
        def process_video():
            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open the video file.")
                progress_window.destroy()
                return

            name = self.person_name.get().strip()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_process = min(total_frames, 100)  # Process max 100 frames
            step = max(1, total_frames // frames_to_process)

            faces_added = 0

            if self.using_insightface:
                try:
                    unique_embeddings = set()  # To track unique faces

                    for i in range(0, total_frames, step):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()

                        if not ret:
                            break

                        # Detect faces
                        faces = self.face_analyzer.get(frame)

                        for face in faces:
                            # Get face embedding
                            embedding = face.embedding

                            # Convert to tuple for hashing (to check uniqueness)
                            embedding_tuple = tuple(embedding)

                            # Check if this is a unique face (rough deduplication)
                            is_unique = True
                            for existing_embedding in unique_embeddings:
                                dist = cosine(embedding, np.array(existing_embedding))
                                if dist < 0.2:  # If very similar to an existing face
                                    is_unique = False
                                    break

                            if is_unique:
                                unique_embeddings.add(embedding_tuple)

                                # Add to database
                                if name not in self.face_database:
                                    self.face_database[name] = []

                                self.face_database[name].append(embedding)
                                faces_added += 1

                                # Save a sample frame with this face
                                x1, y1, x2, y2 = map(int, face.bbox)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                os.makedirs(os.path.join(FACE_DB_PATH, "images"), exist_ok=True)
                                timestamp = int(time.time())
                                img_path = os.path.join(FACE_DB_PATH, "images", f"{name}_{timestamp}_{faces_added}.jpg")
                                cv2.imwrite(img_path, frame)

                except Exception as e:
                    # Use fallback method if InsightFace fails
                    print(f"InsightFace error in video processing: {str(e)}")
                    faces_added = self.process_video_fallback(cap, name, step, total_frames)
            else:
                # Use fallback method
                faces_added = self.process_video_fallback(cap, name, step, total_frames)

            # Save database
            self.save_face_database()

            cap.release()

            # Show result in the main thread
            self.root.after(0, lambda: messagebox.showinfo("Success",
                                                           f"{faces_added} unique face(s) for {name} added from video."))
            self.root.after(0, progress_window.destroy)

        # Start processing in a separate thread
        threading.Thread(target=process_video, daemon=True).start()

    def process_video_fallback(self, cap, name, step, total_frames):
        """Process video using fallback method"""
        unique_face_encodings = []
        faces_added = 0

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                break

            # Detect faces using fallback method
            faces = self.detect_faces_fallback(frame)

            for (x, y, w, h) in faces:
                # Get face embedding
                embedding = self.get_face_embedding_fallback(frame, (x, y, w, h))

                # Check if this is a unique face
                is_unique = True
                for existing_encoding in unique_face_encodings:
                    dist = cosine(embedding, existing_encoding)
                    if dist < 0.4:  # If similar to an existing face (more permissive threshold)
                        is_unique = False
                        break

                if is_unique:
                    unique_face_encodings.append(embedding)

                    # Add to database
                    if name not in self.face_database:
                        self.face_database[name] = []

                    self.face_database[name].append(embedding)
                    faces_added += 1

                    # Save a sample frame with this face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    os.makedirs(os.path.join(FACE_DB_PATH, "images"), exist_ok=True)
                    timestamp = int(time.time())
                    img_path = os.path.join(FACE_DB_PATH, "images", f"{name}_{timestamp}_{faces_added}.jpg")
                    cv2.imwrite(img_path, frame)

        return faces_added

    def view_face_database(self):
        """View the current face database"""
        if not self.face_database:
            messagebox.showinfo("Database", "The face database is currently empty.")
            return

        # Create a new window to show the database
        db_window = tk.Toplevel(self.root)
        db_window.title("Face Database")
        db_window.geometry("400x400")

        # Create a scrollable text area
        text_frame = ttk.Frame(db_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_area = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=text_area.yview)

        # Add database info to the text area
        text_area.insert(tk.END, "Face Database Contents:\n\n")

        total_faces = 0
        for name, embeddings in self.face_database.items():
            num_faces = len(embeddings)
            total_faces += num_faces
            text_area.insert(tk.END, f"{name}: {num_faces} face(s)\n")

        text_area.insert(tk.END, f"\nTotal: {len(self.face_database)} people, {total_faces} faces")
        text_area.config(state=tk.DISABLED)  # Make read-only

    def clear_face_database(self):
        """Clear the entire face database"""
        if not self.face_database:
            messagebox.showinfo("Database", "The face database is already empty.")
            return

        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm",
                                      "Are you sure you want to clear the entire face database? This cannot be undone.")

        if confirm:
            self.face_database = {}
            self.save_face_database()
            messagebox.showinfo("Success", "Face database has been cleared.")

    def start_accuracy_test(self):
        """Start the accuracy testing mode"""
        if not self.is_running:
            messagebox.showwarning("Warning", "Please start the camera first.")
            return

        if not self.face_database:
            messagebox.showwarning("Warning", "Face database is empty. Add faces before testing.")
            return

        # Reset metrics
        self.reset_accuracy_metrics()

        # Just switch to recognition mode
        self.recognition_mode.set("Recognition")

        messagebox.showinfo("Accuracy Test",
                            "Accuracy test started. Metrics will be collected and displayed in real-time.")

    def reset_accuracy_metrics(self):
        """Reset all accuracy metrics"""
        self.accuracy_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "total_frames": 0
        }

    def update_stats(self):
        """Update the statistics display"""
        # Calculate metrics
        tp = self.accuracy_metrics["true_positives"]
        fp = self.accuracy_metrics["false_positives"]
        fn = self.accuracy_metrics["false_negatives"]
        total = self.accuracy_metrics["total_frames"]

        if tp + fp > 0:
            precision = tp / (tp + fp) * 100
        else:
            precision = 0

        if tp + fn > 0:
            recall = tp / (tp + fn) * 100
        else:
            recall = 0

        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0

        # Update stats text
        stats_text = f"Database Size: {len(self.face_database)} people\n\n"
        stats_text += f"Accuracy Metrics:\n"
        stats_text += f"True Positives: {tp}\n"
        stats_text += f"False Positives: {fp}\n"
        stats_text += f"False Negatives: {fn}\n"
        stats_text += f"Total Frames: {total}\n\n"
        stats_text += f"Precision: {precision:.2f}%\n"
        stats_text += f"Recall: {recall:.2f}%\n"
        stats_text += f"F1 Score: {f1_score:.2f}\n\n"

        # Add model info
        stats_text += "Current Models:\n"
        if self.using_insightface:
            stats_text += "✓ InsightFace\n"
        else:
            stats_text += "✗ InsightFace\n"

        if self.using_mediapipe:
            stats_text += "✓ MediaPipe\n"
        else:
            stats_text += "✗ MediaPipe\n"

        stats_text += "✓ OpenCV (fallback)\n\n"

        if self.is_running:
            current_mode = self.recognition_mode.get()
            stats_text += f"Current Mode: {current_mode}"

        self.stats_display.config(text=stats_text)

        # Schedule the next update
        self.root.after(1000, self.update_stats)


def main():
    # Create the Tkinter app
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
