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
FACE_RECOGNITION_THRESHOLD = 0.3  # Lower value = stricter matching

# OpenCV Haar Cascade as fallback
CASCADE_PATH = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        
        # Apply dark theme
        self.setup_dark_theme()
        
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
                                     foreground="#3498DB", font=("Arial", 10, "italic"))
        self.status_label.pack(anchor=tk.W, pady=(5, 10))
        
        # Start model initialization in background
        threading.Thread(target=self.setup_models, daemon=True).start()
        
    def setup_dark_theme(self):
        """Setup dark mode theme for the application"""
        # Dark mode color palette
        self.colors = {
            "bg": "#1E1E1E",           # Dark background
            "fg": "#E0E0E0",           # Light text
            "accent": "#3498DB",       # Blue accent
            "accent_dark": "#2980B9",  # Darker blue for hover/active states
            "secondary_bg": "#2D2D2D", # Slightly lighter background
            "border": "#3D3D3D",       # Medium gray for borders
            "success": "#2ECC71",      # Green for success
            "warning": "#F39C12",      # Orange for warning
            "error": "#E74C3C"         # Red for error
        }
        
        # Configure root window colors
        self.root.configure(bg=self.colors["bg"])
        
        # Configure ttk styles
        self.style = ttk.Style()
        
        # Try to use a modern theme as base
        try:
            self.style.theme_use("clam")  # Use clam as it's more customizable
        except:
            pass  # Fallback to default if clam is not available
            
        # Configure colors for all ttk widgets
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["fg"])
        self.style.configure("TButton", 
                            background=self.colors["secondary_bg"], 
                            foreground=self.colors["fg"],
                            bordercolor=self.colors["border"],
                            relief="flat",
                            focuscolor=self.colors["accent"])
        
        self.style.map("TButton",
                     background=[("active", self.colors["accent"]), 
                                ("pressed", self.colors["accent_dark"])],
                     foreground=[("active", "#FFFFFF"), ("pressed", "#FFFFFF")])
        
        self.style.configure("TRadiobutton", 
                           background=self.colors["bg"], 
                           foreground=self.colors["fg"])
        
        self.style.configure("TEntry", 
                           fieldbackground=self.colors["secondary_bg"],
                           foreground=self.colors["fg"],
                           bordercolor=self.colors["border"])
        
        self.style.configure("TCombobox", 
                           fieldbackground=self.colors["secondary_bg"],
                           foreground=self.colors["fg"],
                           bordercolor=self.colors["border"])
        
        self.style.map("TCombobox",
                     fieldbackground=[("readonly", self.colors["secondary_bg"])],
                     foreground=[("readonly", self.colors["fg"])])
        
        # Custom styles
        self.style.configure("Title.TLabel", 
                           background=self.colors["bg"], 
                           foreground=self.colors["accent"],
                           font=("Arial", 12, "bold"))
        
        self.style.configure("Status.TLabel", 
                           background=self.colors["bg"], 
                           foreground=self.colors["accent"],
                           font=("Arial", 10, "italic"))
        
        self.style.configure("Stats.TLabel", 
                           background=self.colors["bg"], 
                           foreground=self.colors["fg"],
                           font=("Courier", 10))
        
        # Override dropdown menu appearance via a hook
        self.root.option_add('*TCombobox*Listbox.background', self.colors["secondary_bg"])
        self.root.option_add('*TCombobox*Listbox.foreground', self.colors["fg"])
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.colors["accent"])
        self.root.option_add('*TCombobox*Listbox.selectForeground', "#FFFFFF")
    
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
        """Detect available camera devices including 3rd party camera software"""
        available_cameras = []
        
        # First try DirectShow/DSHOW backend (better for Windows 3rd party cameras)
        try:
            # On Windows, try to use DirectShow which often works better with 3rd party cameras
            for i in range(20):  # Extended range to 20 to catch more virtual cameras
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        # Try to get camera name
                        try:
                            camera_name = f"Camera {i}: {cap.get(cv2.CAP_PROP_BACKEND_NAME)}"
                        except:
                            camera_name = f"Camera {i} (DirectShow)"
                        available_cameras.append((camera_name, i))
                    cap.release()
        except Exception as e:
            print(f"DirectShow camera detection error: {e}")
        
        # If no cameras were found with DirectShow, try default backend
        if not available_cameras:
            for i in range(20):  # Extended range to catch more cameras
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        try:
                            camera_name = f"Camera {i}: {cap.getBackendName()}"
                        except:
                            camera_name = f"Camera {i}"
                        available_cameras.append((camera_name, i))
                    cap.release()
        
        # Add DroidCam specific option
        available_cameras.append(("DroidCam (Phone Camera)", "droidcam"))
                
        # Add custom camera index option
        available_cameras.append(("Custom camera index...", -2))
        
        # If still no cameras found, add a placeholder
        if len(available_cameras) <= 2:  # Only has DroidCam and Custom options
            available_cameras.insert(0, ("No cameras found", -1))
            
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
        """Setup the Tkinter user interface with minimalist dark theme"""
        # Create main frames with adjusted padding for minimalist look
        self.control_frame = ttk.Frame(self.root, padding=15)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.video_frame = ttk.Frame(self.root, padding=5)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.stats_frame = ttk.Frame(self.root, padding=15, width=180)
        self.stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Add separator line between sections for cleaner look
        separator = ttk.Separator(self.root, orient=tk.VERTICAL)
        separator.pack(side=tk.RIGHT, fill=tk.Y, padx=0, pady=5, before=self.stats_frame)
        
        # Add app title at the top
        ttk.Label(self.control_frame, text=APP_TITLE, style="Title.TLabel").pack(anchor=tk.W, pady=(0, 15))
        
        # Camera controls section - simplified
        ttk.Label(self.control_frame, text="CAMERA", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        # Camera selection dropdown
        self.camera_combo = ttk.Combobox(self.control_frame, state="readonly", width=25)
        self.camera_combo['values'] = [cam[0] for cam in self.available_cameras]
        self.camera_combo.current(0)
        self.camera_combo.pack(anchor=tk.W, pady=(0, 10), fill=tk.X)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change)
        
        # Start/Stop button
        self.start_stop_btn = ttk.Button(self.control_frame, text="Start Camera", command=self.toggle_camera)
        self.start_stop_btn.pack(anchor=tk.W, pady=(0, 20), fill=tk.X)
        
        # Add separator
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Mode selection
        ttk.Label(self.control_frame, text="MODE", style="Title.TLabel").pack(anchor=tk.W, pady=(10, 10))
        ttk.Radiobutton(self.control_frame, text="Recognition", variable=self.recognition_mode, value="Recognition").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(self.control_frame, text="Training", variable=self.recognition_mode, value="Training").pack(anchor=tk.W, pady=(0, 10))
        
        # Person name entry (for training)
        ttk.Label(self.control_frame, text="Person Name").pack(anchor=tk.W)
        ttk.Entry(self.control_frame, textvariable=self.person_name).pack(anchor=tk.W, fill=tk.X, pady=(0, 15))
        
        # Add separator
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Training controls
        ttk.Label(self.control_frame, text="TRAINING", style="Title.TLabel").pack(anchor=tk.W, pady=(10, 10))
        
        # Add face buttons
        ttk.Button(self.control_frame, text="Add Face from Camera", command=self.add_face_from_camera).pack(anchor=tk.W, pady=(0, 5), fill=tk.X)
        ttk.Button(self.control_frame, text="Add Face from Image", command=self.add_face_from_image).pack(anchor=tk.W, pady=(0, 5), fill=tk.X)
        ttk.Button(self.control_frame, text="Add Face from Video", command=self.add_face_from_video).pack(anchor=tk.W, pady=(0, 15), fill=tk.X)
        
        # Add separator
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Database management
        ttk.Label(self.control_frame, text="DATABASE", style="Title.TLabel").pack(anchor=tk.W, pady=(10, 10))
        ttk.Button(self.control_frame, text="View Database", command=self.view_face_database).pack(anchor=tk.W, pady=(0, 5), fill=tk.X)
        ttk.Button(self.control_frame, text="Clear Database", command=self.clear_face_database).pack(anchor=tk.W, pady=(0, 15), fill=tk.X)
        
        # Add separator
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Accuracy testing controls
        ttk.Label(self.control_frame, text="TESTING", style="Title.TLabel").pack(anchor=tk.W, pady=(10, 10))
        ttk.Button(self.control_frame, text="Start Accuracy Test", command=self.start_accuracy_test).pack(anchor=tk.W, pady=(0, 5), fill=tk.X)
        ttk.Button(self.control_frame, text="Reset Metrics", command=self.reset_accuracy_metrics).pack(anchor=tk.W, pady=(0, 5), fill=tk.X)
        
        # Video display area with dark border
        video_container = ttk.Frame(self.video_frame, style="VideoFrame.TFrame")
        video_container.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_container, background=self.colors["bg"])
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Stats display area
        ttk.Label(self.stats_frame, text="STATISTICS", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        self.stats_display = ttk.Label(self.stats_frame, text="No data available", 
                                     justify=tk.LEFT, style="Stats.TLabel")
        self.stats_display.pack(anchor=tk.W, fill=tk.X)
        
        # Create a custom style for the video frame with border
        self.style.configure("VideoFrame.TFrame", 
                           background=self.colors["bg"],
                           borderwidth=1,
                           relief="solid")
        
        # Start stats updating
        self.update_stats()
    
    def on_camera_change(self, event):
        """Handle camera selection change"""
        if self.is_running:
            self.toggle_camera()  # Stop the current camera
        
        # Get the selected camera index
        selected_idx = self.camera_combo.current()
        if 0 <= selected_idx < len(self.available_cameras):
            camera_value = self.available_cameras[selected_idx][1]
            
            # Check if this is the custom camera option
            if camera_value == -2:
                self.prompt_custom_camera_index()
            else:
                self.camera_index = camera_value
    
    def prompt_custom_camera_index(self):
        """Prompt the user to enter a custom camera index"""
        # Create a simple dialog to enter camera index
        dialog = tk.Toplevel(self.root)
        dialog.title("Custom Camera Index")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()  # Make dialog modal
        
        ttk.Label(dialog, text="Enter camera index or URL:").pack(pady=(20, 5))
        
        # Variable to store result
        result_var = tk.StringVar()
        entry = ttk.Entry(dialog, textvariable=result_var, width=30)
        entry.pack(padx=20, pady=5)
        entry.focus_set()
        
        # Helper text
        ttk.Label(dialog, text="Examples: 0, 1, 2 or rtsp://...", 
                 font=("", 8, "italic")).pack()
        
        def on_ok():
            value = result_var.get().strip()
            try:
                # Check if it's an integer
                if value.isdigit():
                    self.camera_index = int(value)
                # Check if it's a URL string
                elif value.startswith(("http://", "https://", "rtsp://", "rtmp://")):
                    self.camera_index = value
                else:
                    messagebox.showwarning("Invalid Input", "Please enter a valid camera index or URL")
                    return
                
                # Test camera connection
                self.status_var.set(f"Testing camera connection to {value}...")
                cap = cv2.VideoCapture(self.camera_index)
                if not cap.isOpened():
                    self.status_var.set("Failed to connect to camera")
                    messagebox.showwarning("Connection Failed", 
                                          f"Could not connect to camera at {value}.\nThe camera might be in use by another application.")
                    cap.release()
                    return
                
                # If successful, close the dialog
                cap.release()
                self.status_var.set(f"Connected to custom camera: {value}")
                dialog.destroy()
                
                # Update the combobox to show a custom entry
                combo_values = list(self.camera_combo['values'])
                # Find the "Custom camera index..." option and replace its display name
                for i, val in enumerate(combo_values):
                    if "Custom camera" in val:
                        combo_values[i] = f"Custom camera: {value}"
                        break
                
                self.camera_combo['values'] = tuple(combo_values)
                self.camera_combo.current(len(combo_values) - 1)  # Select the custom entry
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to set camera: {str(e)}")
        
        def on_cancel():
            dialog.destroy()
            # Reset to first camera if available
            if len(self.available_cameras) > 0 and self.available_cameras[0][1] >= 0:
                self.camera_combo.current(0)
                self.camera_index = self.available_cameras[0][1]
                
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10, fill=tk.X)
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=(40, 10))
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=(10, 40))
        
        # Handle Enter key
        dialog.bind("<Return>", lambda event: on_ok())
        dialog.bind("<Escape>", lambda event: on_cancel())
    
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
        # Try with DirectShow first on Windows for better 3rd party camera support
        try:
            if isinstance(self.camera_index, int) and self.camera_index >= 0:
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                
                # If DirectShow fails, fall back to default
                if not self.cap.isOpened():
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.camera_index)
            else:
                # For URL cameras or special indices
                self.cap = cv2.VideoCapture(self.camera_index)
        except Exception as e:
            print(f"Error with DirectShow, using default backend: {e}")
            self.cap = cv2.VideoCapture(self.camera_index)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            error_message = f"Could not open camera {self.camera_index}"
            
            # Try to provide more helpful error message
            if isinstance(self.camera_index, int) and self.camera_index >= 0:
                error_message += "\n\nPossible causes:\n- Camera is in use by another application\n- Camera driver issues\n- Camera privacy settings"
            elif isinstance(self.camera_index, str):
                error_message += "\n\nPlease check the URL format and network connection."
                
            messagebox.showerror("Camera Error", error_message)
            self.is_running = False
            self.start_stop_btn.config(text="Start Camera")
            return
            
        # Set higher resolution if possible (helpful for face detection)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except:
            pass  # Ignore if setting resolution fails
        
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
        face_region = frame[y:y+h, x:x+w]
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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 165, 0), 2)  # Orange for training mode
            
            # Add label indicating training mode
            cv2.putText(frame, "Training Mode", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
    
    def find_face_match(self, embedding):
        """Find the best match for a face embedding in the database"""
        best_match_name = "Unknown"
        best_match_dist = float('inf')
        
        for name, embeddings_list in self.face_database.items():
            for stored_embedding in embeddings_list:
                # Make sure shapes match for comparison
                if len(embedding) != len(stored_embedding):
                    continue
                    
                # Calculate cosine distance
                dist = cosine(embedding, stored_embedding)
                
                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_name = name
        
        return best_match_name, best_match_dist
    
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
                    largest_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
                    
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
        cv2.rectangle(self.current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
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
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return faces_added
    
    def add_face_from_video(self):
        """Add faces from a video file with improved multi-angle capture"""
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
        
        # Ask for multi-angle training configuration
        config_window = tk.Toplevel(self.root)
        config_window.title("Multi-Angle Training Configuration")
        config_window.geometry("400x250")
        
        # Similarity threshold
        similarity_var = tk.DoubleVar(value=0.4)  # Higher value = capture more angles
        ttk.Label(config_window, text="Angle Diversity (lower = more diverse angles):").pack(anchor=tk.W, padx=10, pady=(10, 0))
        ttk.Scale(config_window, from_=0.2, to=0.6, variable=similarity_var, orient=tk.HORIZONTAL, length=300).pack(padx=10, pady=(0, 10))
        ttk.Label(config_window, text="0.2 (Many angles) ←→ 0.6 (Fewer angles)").pack(padx=10)
        
        # Frame sampling rate
        frame_sampling_var = tk.IntVar(value=5)
        ttk.Label(config_window, text="Frame sampling rate (lower = more frames):").pack(anchor=tk.W, padx=10, pady=(10, 0))
        ttk.Scale(config_window, from_=1, to=10, variable=frame_sampling_var, orient=tk.HORIZONTAL, length=300).pack(padx=10, pady=(0, 10))
        ttk.Label(config_window, text="1 (Many frames) ←→ 10 (Fewer frames)").pack(padx=10)
        
        # Progress variable
        progress_var = tk.StringVar(value="Ready to process")
        process_btn = ttk.Button(config_window, text="Start Processing")
        process_btn.pack(pady=10)
        ttk.Label(config_window, textvariable=progress_var).pack(pady=5)
        
        # Function to process video in the background
        def process_video():
            process_btn.config(state=tk.DISABLED)
            progress_var.set("Opening video...")
            
            # Get configuration values
            similarity_threshold = similarity_var.get()
            frame_sampling = max(1, frame_sampling_var.get())
            
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open the video file.")
                config_window.destroy()
                return
            
            name = self.person_name.get().strip()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # More intelligent frame sampling based on video length
            video_length_seconds = total_frames / fps
            frames_to_process = int(total_frames / frame_sampling)
            
            progress_var.set(f"Processing {frames_to_process} frames from {video_length_seconds:.1f}s video...")
            
            faces_added = 0
            angle_categories = {}  # To categorize faces by angle
            
            if self.using_insightface:
                try:
                    # Process the video frames
                    for i in range(0, total_frames, frame_sampling):
                        if i % 20 == 0:  # Update progress every 20 frames
                            progress_percentage = min(100, int((i / total_frames) * 100))
                            progress_var.set(f"Processing... {progress_percentage}% ({faces_added} faces found)")
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        
                        if not ret:
                            break
                        
                        # Detect faces
                        faces = self.face_analyzer.get(frame)
                        
                        if not faces:
                            continue
                            
                        # Use the face with the largest area if multiple are detected
                        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        
                        # Get face embedding
                        embedding = face.embedding
                        
                        # Skip if no embedding
                        if embedding is None:
                            continue
                        
                        # Find which angle category this face belongs to (if any)
                        best_category = None
                        best_similarity = -1
                        
                        for category, rep_embedding in angle_categories.items():
                            # Calculate similarity
                            similarity = 1 - cosine(embedding, rep_embedding)
                            
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_category = category
                        
                        # If similar to existing category, skip
                        if best_category is not None and best_similarity > similarity_threshold:
                            continue
                            
                        # Create new category
                        new_category = f"angle_{len(angle_categories) + 1}"
                        angle_categories[new_category] = embedding
                        
                        # Add to database
                        if name not in self.face_database:
                            self.face_database[name] = []
                        
                        self.face_database[name].append(embedding)
                        faces_added += 1
                        
                        # Save a sample frame with this face
                        x1, y1, x2, y2 = map(int, face.bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Angle {len(angle_categories)}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        os.makedirs(os.path.join(FACE_DB_PATH, "images"), exist_ok=True)
                        img_path = os.path.join(FACE_DB_PATH, "images", f"{name}_angle{len(angle_categories)}.jpg")
                        cv2.imwrite(img_path, frame)
                except Exception as e:
                    # Use fallback method if InsightFace fails
                    progress_var.set(f"InsightFace error: {str(e)}. Using fallback method...")
                    faces_added = self.process_video_fallback(cap, name, frame_sampling, total_frames)
            else:
                # Use fallback method
                progress_var.set("InsightFace unavailable. Using fallback method...")
                faces_added = self.process_video_fallback(cap, name, frame_sampling, total_frames)
            
            # Save database
            self.save_face_database()
            
            cap.release()
            
            # Show result
            progress_var.set(f"Completed! {faces_added} unique face angles captured.")
            
            # Enable closing the window
            close_btn = ttk.Button(config_window, text="Close", command=config_window.destroy)
            close_btn.pack(pady=10)
            
            # Update main window stats
            self.update_stats()
        
        # Set button action
        process_btn.config(command=lambda: threading.Thread(target=process_video, daemon=True).start())
        
        # Center window
        config_window.update_idletasks()
        width = config_window.winfo_width()
        height = config_window.winfo_height()
        x = (config_window.winfo_screenwidth() // 2) - (width // 2)
        y = (config_window.winfo_screenheight() // 2) - (height // 2)
        config_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        config_window.focus_set()
    
    def process_video_fallback(self, cap, name, step, total_frames):
        """Process video using fallback method with improved angle detection"""
        angle_categories = {}  # For tracking different face angles
        faces_added = 0
        
        # Process frames at regular intervals
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect faces using fallback method
            faces = self.detect_faces_fallback(frame)
            
            if not faces:
                continue
                
            # Use largest face if multiple detected
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Get face embedding
            embedding = self.get_face_embedding_fallback(frame, (x, y, w, h))
            
            # Find which angle category this face belongs to (if any)
            best_category = None
            best_similarity = -1
            
            for category, rep_embedding in angle_categories.items():
                # Calculate similarity
                similarity = 1 - cosine(embedding, rep_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category = category
            
            # Similarity threshold - adjust for better angle diversity
            similarity_threshold = 0.5  # More permissive for fallback method
            
            # If similar to existing category, skip
            if best_category is not None and best_similarity > similarity_threshold:
                continue
                
            # Create new category
            new_category = f"angle_{len(angle_categories) + 1}"
            angle_categories[new_category] = embedding
            
            # Add to database
            if name not in self.face_database:
                self.face_database[name] = []
            
            self.face_database[name].append(embedding)
            faces_added += 1
            
            # Save a sample frame with this face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Angle {len(angle_categories)}", 
                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            os.makedirs(os.path.join(FACE_DB_PATH, "images"), exist_ok=True)
            img_path = os.path.join(FACE_DB_PATH, "images", f"{name}_angle{len(angle_categories)}.jpg")
            cv2.imwrite(img_path, frame)
        
        return faces_added
    
    def view_face_database(self):
        """View the current face database with dark mode styling"""
        if not self.face_database:
            messagebox.showinfo("Database", "The face database is currently empty.")
            return
        
        # Create a new window to show the database
        db_window = tk.Toplevel(self.root)
        db_window.title("Face Database")
        db_window.geometry("400x400")
        db_window.configure(bg=self.colors["bg"])
        
        # Apply icon if available
        try:
            db_window.iconbitmap(self.root.iconbitmap())
        except:
            pass
            
        # Create a scrollable text area
        text_frame = ttk.Frame(db_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_area = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                           bg=self.colors["secondary_bg"], fg=self.colors["fg"],
                           insertbackground=self.colors["fg"], 
                           selectbackground=self.colors["accent"],
                           selectforeground="#FFFFFF", 
                           borderwidth=0, padx=10, pady=10)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=text_area.yview)
        
        # Add database info to the text area with minimalist style
        text_area.insert(tk.END, "FACE DATABASE\n\n", "title")
        
        total_faces = 0
        for name, embeddings in self.face_database.items():
            num_faces = len(embeddings)
            total_faces += num_faces
            text_area.insert(tk.END, f"{name}: ", "name")
            text_area.insert(tk.END, f"{num_faces} face(s)\n", "count")
        
        text_area.insert(tk.END, f"\nTOTAL: {len(self.face_database)} people, {total_faces} faces", "total")
        
        # Configure text tags
        text_area.tag_configure("title", font=("Arial", 12, "bold"), foreground=self.colors["accent"])
        text_area.tag_configure("name", font=("Arial", 10, "bold"))
        text_area.tag_configure("count", font=("Arial", 10))
        text_area.tag_configure("total", font=("Arial", 10, "bold"), foreground=self.colors["accent"])
        
        text_area.config(state=tk.DISABLED)  # Make read-only
        
        # Add a close button with custom style
        close_btn = ttk.Button(db_window, text="Close", command=db_window.destroy)
        close_btn.pack(pady=15)
        
        # Center window on parent
        db_window.update_idletasks()
        width = db_window.winfo_width()
        height = db_window.winfo_height()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (width // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (height // 2)
        db_window.geometry(f'{width}x{height}+{x}+{y}')
    
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
        """Update the statistics display with dark mode styling"""
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
        
        # Update stats text - minimalist format
        stats_text = f"PEOPLE: {len(self.face_database)}\n\n"
        
        # Metrics section
        stats_text += "METRICS\n"
        stats_text += f"True+:  {tp}\n"
        stats_text += f"False+: {fp}\n"
        stats_text += f"False-: {fn}\n"
        stats_text += f"Frames: {total}\n\n"
        
        # Accuracy section
        stats_text += "ACCURACY\n"
        stats_text += f"Precision: {precision:.1f}%\n"
        stats_text += f"Recall:    {recall:.1f}%\n"
        stats_text += f"F1 Score:  {f1_score:.1f}\n\n"
        
        # Add model info
        stats_text += "MODELS\n"
        if self.using_insightface:
            stats_text += "✓ InsightFace\n"
        else:
            stats_text += "✗ InsightFace\n"
            
        if self.using_mediapipe:
            stats_text += "✓ MediaPipe\n"
        else:
            stats_text += "✗ MediaPipe\n"
            
        stats_text += "✓ OpenCV\n\n"
        
        if self.is_running:
            current_mode = self.recognition_mode.get()
            stats_text += f"MODE: {current_mode}"
        
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
