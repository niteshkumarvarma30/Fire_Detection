🔥 Real-Time Fire Detection System using YOLOv5
A high-precision, real-time fire detection system built using Computer Vision and Deep Learning. This project leverages a custom-trained YOLOv5 model to identify fire in live webcam feeds, trigger an audible alarm, and log evidence for safety monitoring.

🌟 Key Features
Custom Trained Model: Leveraged Transfer Learning on a massive dataset of 20,000+ images to ensure high recall and precision.

Real-Time Inference: Optimized for low-latency detection directly from a webcam or IP camera feed.

Smart Alert System: Uses Multithreading to trigger a sound alert (alert.mp3) without interrupting the video processing loop.

Automated Forensics: Automatically saves timestamped frames of detected fire into a fire_frames/ directory for post-incident analysis.

Optimized Performance: Implements a 5-second alert interval to prevent redundant alarms while maintaining continuous monitoring.

🛠️ Tech Stack
Language: Python

Deep Learning: PyTorch, YOLOv5

Computer Vision: OpenCV

Tools: NumPy, Threading, Playsound

🚀 Getting Started
1. Installation
Clone the repository and install the dependencies:

Bash
git clone https://github.com/yourusername/fire-detection-yolov5.git
cd fire-detection-yolov5
pip install -r requirements.txt
2. File Setup
Ensure the following files are in the root directory:

fire.pt: custom-trained weights (20k images).

alert.mp3: The sound file for the alarm.

3. Run the System


Bash
python main.py


📊 Dataset & Training
The model was developed using Transfer Learning on a pre-trained YOLOv5 backbone.

Total Images: 20,000+ (Fire and Smoke classes).

Augmentation: Applied flip, rotation, and brightness adjustments to improve robustness in varying lighting conditions.

Training Hardware: Trained on NVIDIA GPU RTX 3050

Accuracy: Achieved a Mean Average Precision (mAP) of 0.92.

📂 Project Structure
Plaintext
fire-detection/
├── fire_frames/       # Auto-saved images of detected fire
├── fire.pt            # Custom trained weights
├── main.py            # Main execution script
├── alert.mp3          # Alarm sound
└── requirements.txt   # Dependencies
🔮 Future Roadmap
[ ] Integration with a Telegram Bot for mobile notifications.

[ ] Smoke detection enhancement for early warning.

[ ] Deployment on Edge devices like Jetson Nano.

🤝 Contributing
Contributions are welcome! If you have ideas for improving the detection logic or adding new features, feel free to fork this repo and submit a PR.
