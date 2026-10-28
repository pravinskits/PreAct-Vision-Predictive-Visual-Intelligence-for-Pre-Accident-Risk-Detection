# 👁️ PreAct-Vision: Predictive Visual Intelligence for Proactive Hazard Detection

![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## The Paradigm Shift: From Reactive to Proactive
Traditional computer vision applications are entirely **reactive**. They draw bounding boxes around pedestrians, vehicles, and objects *as* events happen. By the time a risk is flagged, a collision or accident might have already occurred.

**PreAct-Vision flips this paradigm.** Instead of merely detecting, we track movement vectors, forecast trajectories, and calculate Time-to-Collision (TTC) to proactively alert systems of impending hazards *before* they manifest.

## 🏗️ Architectural Breakdown

The PreAct-Vision engine operates on a robust, real-time 4-layer pipeline:

1. **Perception (Detection Layer)**
   - Utilizes state-of-the-art **YOLOv8** to process video frames, detecting objects/people, and extracting bounding boxes and confidence scores. Hardware acceleration (CUDA) is dynamically allocated for maximum FPS.
2. **Tracking (State Estimation Layer)**
   - A proprietary centroid-based tracking algorithm assigns persistent IDs across frames. It calculates velocity vectors (`dx, dy`) using Exponential Moving Average (EMA) smoothing to eliminate tracking jitter.
3. **Prediction (Forecasting Layer)**
   - Projects motion vectors 30+ frames into the future, mapping trajectory arcs (visualized as vector arrows) and calculating the Time-to-Collision (TTC) based on speed and relative positioning.
4. **Warn (Risk Assessment Layer)**
   - A rule-based risk evaluation engine parses TTC metrics and lateral speed into actionable alerts (Green = Low Risk, Yellow = Caution, Red = High Collision Risk).

## 🛠️ Tech Stack & Requirements

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, Ultralytics YOLOv8
- **Computer Vision:** OpenCV (`opencv-python`)
- **Mathematics:** NumPy
- **User Interface:** Streamlit

## 🚀 Installation & Quickstart

1. **Clone the repository and navigate to the root directory:**
   ```bash
   git clone https://github.com/yourusername/PreAct-Vision.git
   cd PreAct-Vision
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Command Line Pipeline:**
   ```bash
   python main.py --input path/to/video.mp4 --output result.mp4
   ```
   *(To use your webcam, run: `python main.py --input 0`)*

4. **Launch the Interactive Web Dashboard:**
   *(If the `streamlit` command is not recognized directly in your terminal, use the Python module executor!)*
   ```bash
   python -m streamlit run app.py
   ```

## 🌍 Real-World Use Cases

- **Industrial & Warehouse Safety:** Alerting forklift drivers of blind-corner pedestrian traffic before physical line-of-sight is established.
- **Smart Infrastructure (V2X):** Monitoring traffic intersections to broadcast hazard warnings to connected vehicles if a collision trajectory is detected.
- **Public Spaces & Retail:** Crowd control monitoring to anticipate surges or structural bottlenecks based on pedestrian movement forecasting.

---
*Built for the future of proactive safety systems.*
