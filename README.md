# Face Recognition System

A powerful, real-time face detection and recognition system with an intuitive dark-themed UI, optimized for performance and accuracy testing.

![Face Recognition System](./screenshots/app_screenshot.png)

## Features

- **Real-time face detection and recognition** with multiple backend options
- **High-performance mode** optimized for up to 30+ FPS
- **Brightness and luminance control** for optimal recognition in different lighting conditions
- **Multiple camera support**:
  - Built-in webcams
  - USB cameras
  - IP cameras
  - DroidCam (use your phone as a camera)
  - RTSP streams
- **Training capabilities**:
  - Add faces from camera
  - Add faces from image files
  - Add faces from video files with multi-angle captures
- **Face database management**
- **Accuracy testing** with metrics tracking
- **Modern dark-themed UI** with scrolling status messages

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam, phone camera (via DroidCam), or other compatible camera device
- At least 4GB of RAM (8GB recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-recognition-system.git
   cd face-recognition-system
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python main.py
   ```

## Usage Guide

### Camera Setup

1. Select a camera from the dropdown menu
2. For DroidCam:
   - Install the DroidCam app on your phone
   - Configure IP address and port
   - Select resolution and performance options
3. Click "Start Camera" to begin

### Face Recognition

1. Make sure "Recognition" mode is selected
2. Position your face in the camera view
3. The system will detect and attempt to recognize faces
4. Recognized faces are highlighted in green with name and confidence percentage
5. Unknown faces are highlighted in red

### Training Mode

1. Select "Training" mode
2. Enter a person name
3. Choose one of the following methods:
   - **From Camera**: Captures the current frame
   - **From Image**: Uploads an image file
   - **From Video**: Extracts multiple angles from a video file

### Brightness Control

- Use the + and - buttons to adjust brightness
- Use keyboard shortcuts: B to increase, N to decrease, R to reset
- Monitor the luminance value on the video feed

### Accuracy Testing

1. Train the system with faces
2. Click "Start Accuracy Test"
3. View real-time metrics in the statistics panel:
   - True positives
   - False positives
   - False negatives
   - Precision
   - Recall
   - F1 Score

## Keyboard Shortcuts

- **B**: Increase brightness
- **N**: Decrease brightness
- **R**: Reset brightness to default (100%)

## Configuration

The system automatically creates and uses the following directories:

- `face_database/`: Stores face embeddings and training images
- `models/`: Stores downloaded AI models

## Performance Tips

- For maximum FPS:
  - Use 160x120 resolution with DroidCam
  - Enable ultra-performance mode
  - Use lightweight detection
  - Process every 5-10 frames
- For best recognition accuracy:
  - Use higher resolution
  - Disable lightweight detection
  - Process every frame
  - Train with multiple angles per person

## Advanced Features

### Multi-angle Training

When adding faces from video, the system intelligently extracts different face angles for better recognition accuracy in varied poses and lighting conditions.

### Luminance Measurement

The system measures scene luminance and displays it with a visual indicator, helping you optimize lighting conditions for better face detection.

## Troubleshooting

- **Camera not found**: Check connections, privacy settings, and try a different camera index
- **DroidCam connection issues**: Verify IP address, port, and that both devices are on the same network
- **Low FPS**: Try lower resolution, enable performance mode, or use lightweight detection
- **Poor recognition**: Train with more face angles and adjust lighting for better luminance
