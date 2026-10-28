import streamlit as st
import cv2
import tempfile
import time
import yt_dlp
import threading
import numpy as np
from perception.detector import ObjectDetector
from tracking.tracker import MultiObjectTracker
from prediction.forecaster import TrajectoryForecaster
from risk_assessment.assessor import RiskAssessor
from utils.visualizer import Visualizer

class VideoStreamBuffer:
    """
    Thread-safe video capture buffer for live webcam and network streams.
    Permanently eliminates stuttering and thread conflicts:
    - Protects frame reads, updates, and connection status using threading.Lock().
    - Eliminates race conditions between the background capture thread and the main Streamlit thread.
    - Features a resilient auto-reconnect update loop to smoothly absorb dropped chunks and empty frames.
    - Yields non-blocking access to the latest cached frame without UI freezes.
    """
    def __init__(self, src=0, resize_dims=(640, 360), reconnect_delay=0.5, max_reconnect_attempts=30, max_consecutive_drops=10):
        self.src = src
        self.resize_dims = resize_dims
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.max_consecutive_drops = max_consecutive_drops
        
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.stopped = False
        self.grabbed = False
        self.frame = None
        self.frame_id = 0
        
        # Open capture device safely with zero-buffering
        self.cap = self._open_capture()
        if self.cap is not None and self.cap.isOpened():
            self._configure_capture(self.cap)
            grabbed, initial_frame = self.cap.read()
            if grabbed and initial_frame is not None and initial_frame.size > 0:
                if self.resize_dims:
                    initial_frame = cv2.resize(initial_frame, self.resize_dims)
                with self.cond:
                    self.grabbed = True
                    self.frame = initial_frame
                    self.frame_id += 1

        # Dedicated background daemon thread for continuous non-blocking frame retrieval
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def _open_capture(self):
        try:
            if isinstance(self.src, int) or (isinstance(self.src, str) and str(self.src).isdigit()):
                # DirectShow backend on Windows eliminates MSMF buffering delay and frame stuttering
                cap = cv2.VideoCapture(int(self.src), cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(self.src))
            else:
                cap = cv2.VideoCapture(self.src)
            return cap
        except Exception:
            return None

    def _configure_capture(self, cap):
        try:
            # Drop OS frame buffering to guarantee strictly real-time live frames
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if isinstance(self.src, int) or (isinstance(self.src, str) and str(self.src).isdigit()):
            try:
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            except Exception:
                pass

    def update(self):
        """
        Background capture loop. Reads camera/network frames at hardware speed
        and notifies the consumer thread of fresh frames without queuing delay.
        """
        reconnect_attempts = 0
        consecutive_drops = 0
        
        try:
            while True:
                with self.cond:
                    if self.stopped:
                        break

                # 1. Check if stream capture is open; if not, trigger reconnect
                if self.cap is None or not self.cap.isOpened():
                    reconnect_attempts += 1
                    if self.max_reconnect_attempts and reconnect_attempts > self.max_reconnect_attempts:
                        with self.cond:
                            self.stopped = True
                            self.cond.notify_all()
                        break
                    
                    time.sleep(self.reconnect_delay)
                    with self.cond:
                        if self.stopped:
                            break
                    try:
                        if self.cap is not None:
                            self.cap.release()
                    except Exception:
                        pass
                    self.cap = self._open_capture()
                    if self.cap is not None and self.cap.isOpened():
                        self._configure_capture(self.cap)
                    consecutive_drops = 0
                    continue

                # 2. Read frame directly from hardware/network stream
                try:
                    grabbed, raw_frame = self.cap.read()
                except Exception:
                    grabbed, raw_frame = False, None

                # 3. Handle dropped live stream chunks or empty frames smoothly
                if not grabbed or raw_frame is None or raw_frame.size == 0:
                    consecutive_drops += 1
                    if consecutive_drops <= self.max_consecutive_drops:
                        time.sleep(0.005)
                        continue
                    
                    # Persistent drop: trigger stream auto-reconnection
                    reconnect_attempts += 1
                    if self.max_reconnect_attempts and reconnect_attempts > self.max_reconnect_attempts:
                        with self.cond:
                            self.stopped = True
                            self.cond.notify_all()
                        break
                    
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    time.sleep(self.reconnect_delay)
                    with self.cond:
                        if self.stopped:
                            break
                    self.cap = self._open_capture()
                    if self.cap is not None and self.cap.isOpened():
                        self._configure_capture(self.cap)
                    consecutive_drops = 0
                    continue

                # Frame successfully captured
                reconnect_attempts = 0
                consecutive_drops = 0

                # Resize to target dimensions (640x360)
                if self.resize_dims and raw_frame is not None:
                    if raw_frame.shape[1] != self.resize_dims[0] or raw_frame.shape[0] != self.resize_dims[1]:
                        raw_frame = cv2.resize(raw_frame, self.resize_dims)

                # Thread-safe update of cached frame under condition
                with self.cond:
                    self.grabbed = True
                    self.frame = raw_frame
                    self.frame_id += 1
                    self.cond.notify_all()

                # For network/YouTube streams, pace capture to 30 FPS to prevent socket exhaustion and TLS timeouts
                is_webcam = isinstance(self.src, int) or (isinstance(self.src, str) and str(self.src).isdigit())
                if not is_webcam:
                    time.sleep(0.033)
        finally:
            with self.cond:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                self.cond.notify_all()

    def read(self, wait_for_new=False, timeout=0.08):
        """
        Thread-safe frame retrieval.
        When wait_for_new=True, synchronizes with the hardware capture thread to yield
        only freshly grabbed frames, eliminating duplicate processing and video sticking.
        """
        with self.cond:
            if wait_for_new:
                cur_id = self.frame_id
                self.cond.wait_for(lambda: self.frame_id > cur_id or self.stopped, timeout=timeout)
            
            if self.stopped and self.frame is None:
                return False, None
            if self.frame is not None:
                return True, self.frame.copy()
            return self.grabbed, None

    def isOpened(self):
        with self.cond:
            if self.stopped:
                return False
            return self.cap is not None and self.cap.isOpened()

    def release(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        if self.thread.is_alive() and threading.current_thread() != self.thread:
            self.thread.join(timeout=1.0)
        with self.cond:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def get(self, propId):
        with self.cond:
            if self.cap is not None and self.cap.isOpened():
                try:
                    return self.cap.get(propId)
                except Exception:
                    return 0.0
            return 0.0

# Backward compatibility alias
ThreadedCamera = VideoStreamBuffer

def release_stream_buffer():
    """Cleanly releases and clears active stream buffers stored in st.session_state."""
    if "video_stream_buffer" in st.session_state and st.session_state.video_stream_buffer is not None:
        try:
            st.session_state.video_stream_buffer.release()
        except Exception:
            pass
        st.session_state.video_stream_buffer = None
    st.session_state.current_stream_source = None
    if "file_stream" in st.session_state and st.session_state.file_stream is not None:
        try:
            st.session_state.file_stream.release()
        except Exception:
            pass
        st.session_state.file_stream = None
    st.session_state.current_file_path = None

def get_or_create_stream_buffer(source, resize_dims=(640, 360)):
    """
    Manages the VideoStreamBuffer inside st.session_state.
    Reuses existing background capture buffer when the source hasn't changed,
    preventing Streamlit's script reruns from resetting the capture and freezing playback.
    """
    buf = st.session_state.get("video_stream_buffer")
    curr_src = st.session_state.get("current_stream_source")

    # If already initialized for this exact source and still alive, reuse existing instance
    if buf is not None and curr_src == source and buf.isOpened():
        return buf

    # Otherwise cleanly release previous buffer and create new one
    release_stream_buffer()

    new_buf = VideoStreamBuffer(source, resize_dims=resize_dims)
    st.session_state.video_stream_buffer = new_buf
    st.session_state.current_stream_source = source
    return new_buf

class YDLSilentLogger:
    """Logger to suppress verbose output while capturing errors and warnings from yt-dlp."""
    def __init__(self):
        self.warnings = []
        self.errors = []

    def debug(self, msg):
        pass

    def warning(self, msg):
        self.warnings.append(msg)

    def error(self, msg):
        self.errors.append(msg)

def extract_youtube_stream(youtube_url: str):
    """
    Extracts direct video stream URL from a YouTube link using yt-dlp.
    Configures JavaScript runtime evaluation (Node.js/Deno/QuickJS) to solve YouTube
    player scripts and anti-bot challenge solving.
    Returns: (stream_url, error_message)
    """
    logger = YDLSilentLogger()

    ydl_opts = {
        "format": "bestvideo/best",
        "quiet": True,
        "noplaylist": True,
        "logger": logger,
        "js_runtimes": {
            "node": {},
            "deno": {},
            "quickjs": {},
        },
        "remote_components": ["ejs:github"],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            stream_url = info_dict.get("url", None) if info_dict else None

        if not stream_url:
            js_warning = next(
                (w for w in logger.warnings if any(k in w.lower() for k in ["javascript", "js", "challenge", "signature", "solver", "bot"])),
                None
            )
            if js_warning:
                return None, f"JavaScript runtime challenge evaluation failed: {js_warning}"
            return None, "Could not extract playable stream URL from the provided YouTube link."

        return stream_url, None

    except Exception as e:
        err_msg = str(e)
        if any(k in err_msg.lower() for k in ["javascript", "js runtime", "challenge", "signature", "solver", "bot", "sign in"]):
            return None, f"YouTube player script / JavaScript evaluation error: {err_msg}"
        return None, f"Error fetching YouTube stream: {err_msg}"

@st.cache_resource
def get_object_detector():
    """Initializes and caches the YOLOv8 ObjectDetector model to prevent repeated reloading across script reruns."""
    return ObjectDetector()

# --- PAGE SETUP ---
st.set_page_config(page_title="PreAct-Vision Dashboard", page_icon="👁️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
/* Dedicated styling for the Live Processing Feed to look like a professional monitoring panel */
[data-testid="stImage"] {
    border: 3px solid #00E676;
    border-radius: 12px;
    padding: 8px;
    background-color: #0A0A0A;
    box-shadow: 0px 0px 25px rgba(0, 230, 118, 0.3);
}
</style>
""", unsafe_allow_html=True)

st.title("👁️ PreAct-Vision: Predictive Visual Intelligence")
st.markdown("""
Welcome to the PreAct-Vision dashboard. Our 4-layer pipeline processes video streams to detect hazards *before* they occur.
1. **Perception**: YOLOv8 object detection (optimized at 640x360 with imgsz=416).
2. **Tracking**: ID assignment and velocity extraction via Centroid tracking.
3. **Prediction**: Trajectory forecasting and Time-to-Collision (TTC) calculations.
4. **Warn (Risk Assessment)**: Rule-based hazard evaluation (Low, Medium, High).
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Tuning Parameters")
conf_thresh = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.05,
    max_value=0.80,
    value=0.15,
    step=0.05,
    help="Detection sensitivity. 0.15 captures all distant, small, or partially visible vehicles and pedestrians."
)

imgsz_choice = st.sidebar.selectbox(
    "🔍 Detection Precision (imgsz)",
    ["640 (High Recall - All Objects)", "416 (Balanced - Fast CPU)"],
    index=0,
    help="640 provides maximum recall for distant, small, or bird's-eye objects. 416 runs faster on older CPUs."
)
selected_imgsz = 640 if "640" in imgsz_choice else 416

high_risk_ttc = st.sidebar.slider("High Risk TTC (frames)", 5, 50, 25, 1)
medium_risk_ttc = st.sidebar.slider("Medium Risk TTC (frames)", 10, 100, 60, 1)

target_fps = st.sidebar.slider(
    "⏱️ Target Playback Speed (FPS)",
    min_value=5,
    max_value=30,
    value=15,
    step=1,
    help="Controls the exact playback speed of the video. 12-15 FPS provides comfortable, human-readable pacing without rushing."
)

inference_mode = st.sidebar.radio(
    "⚡ Inference Acceleration",
    options=["Every Frame (Max Accuracy)", "Every 2nd Frame (High Speed 40+ FPS)"],
    index=0,
    help="On CPU, running YOLO every 2nd frame doubles processing speed while maintaining continuous tracking via dead reckoning."
)

st.sidebar.header("📁 Input Source")
input_source = st.sidebar.selectbox(
    "Select Video Source:",
    [
        "Sample Video (car-detection.mp4)",
        "Multi-Hazard Sample (person-bicycle-car-detection.mp4)",
        "Live Webcam (0)",
        "YouTube Live Stream"
    ]
)

# Release stream buffer if input source type changed
if "last_input_source" not in st.session_state:
    st.session_state.last_input_source = input_source
elif st.session_state.last_input_source != input_source:
    st.session_state.last_input_source = input_source
    release_stream_buffer()
    st.session_state.is_running = False
    st.session_state.paused = False
    st.session_state.last_frame = None

youtube_url = ""
if input_source == "YouTube Live Stream":
    youtube_url = st.sidebar.text_input("Enter YouTube Live Stream URL:")

# Session state initialization for safe execution and pause management
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "paused" not in st.session_state:
    st.session_state.paused = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

def handle_run():
    st.session_state.is_running = True
    st.session_state.paused = False
    st.session_state.frame_count = 0
    st.session_state.prev_time = time.time()
    st.session_state.smoothed_fps = 0.0
    st.session_state.total_high_risks = set()
    st.session_state.loop_logs = []
    st.session_state.last_frame = None
    st.session_state.last_detections = []
    if "file_stream" in st.session_state and st.session_state.file_stream is not None:
        try:
            st.session_state.file_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            pass
    st.session_state.tracker = MultiObjectTracker(max_distance=150)
    st.session_state.forecaster = TrajectoryForecaster(future_frames=30)
    st.session_state.forecaster.frame_height = 360
    st.session_state.forecaster.frame_width = 640
    st.session_state.assessor = RiskAssessor(high_risk_ttc=high_risk_ttc, medium_risk_ttc=medium_risk_ttc)
    st.session_state.visualizer = Visualizer()

def handle_stop():
    st.session_state.is_running = False
    st.session_state.paused = False
    st.session_state.last_frame = None
    st.session_state.frame_count = 0
    release_stream_buffer()

def toggle_pause():
    st.session_state.paused = not st.session_state.paused

col_run, col_stop = st.sidebar.columns(2)
run_clicked = col_run.button("🚀 Run Analysis", key="btn_run", on_click=handle_run, use_container_width=True, help="Start video analysis")
stop_clicked = col_stop.button("⏹️ Stop", key="btn_stop", on_click=handle_stop, use_container_width=True, help="Stop analysis and reset feed")

pause_label = "▶️ Resume Stream" if st.session_state.paused else "⏸️ Pause Stream"
st.sidebar.button(
    pause_label,
    key="pause_resume_btn",
    on_click=toggle_pause,
    disabled=not st.session_state.is_running,
    use_container_width=True,
    help="Toggle pause/resume without restarting the stream"
)

if stop_clicked:
    st.sidebar.info("Analysis stopped.")

# --- ACTIVE PIPELINE EXECUTION ---
if st.session_state.is_running:
    if input_source == "Sample Video (car-detection.mp4)":
        video_path = "car-detection.mp4"
    elif input_source == "Multi-Hazard Sample (person-bicycle-car-detection.mp4)":
        video_path = "person-bicycle-car-detection.mp4"
    elif input_source == "Live Webcam (0)":
        video_path = 0
    elif input_source == "YouTube Live Stream":
        if not youtube_url or not youtube_url.strip():
            st.error("Please enter a valid YouTube URL.")
            st.session_state.is_running = False
            st.stop()
        
        clean_url = youtube_url.strip()
        # Evaluate & extract YouTube stream URL only once per URL to avoid repeated yt-dlp latency
        if (
            st.session_state.get("cached_youtube_input") != clean_url
            or not st.session_state.get("cached_stream_url")
        ):
            with st.spinner("Evaluating YouTube player scripts & extracting stream..."):
                extracted_url, error_msg = extract_youtube_stream(clean_url)
                
            if error_msg or not extracted_url:
                st.error(f"❌ {error_msg or 'Could not extract stream URL from the provided YouTube link.'}")
                st.session_state.is_running = False
                st.stop()
                
            st.session_state.cached_youtube_input = clean_url
            st.session_state.cached_stream_url = extracted_url
            release_stream_buffer()

        video_path = st.session_state.cached_stream_url

    # Initialize / retrieve cached pipeline modules
    detector = get_object_detector()
    detector.conf_threshold = conf_thresh
    
    tracker = st.session_state.get("tracker", MultiObjectTracker(max_distance=150))
    st.session_state.tracker = tracker
    
    forecaster = st.session_state.get("forecaster", TrajectoryForecaster(future_frames=30))
    forecaster.frame_height = 360  # Frame height is normalized to 360 for consistent TTC calculation
    forecaster.frame_width = 640
    st.session_state.forecaster = forecaster
    
    assessor = st.session_state.get("assessor", RiskAssessor(high_risk_ttc=high_risk_ttc, medium_risk_ttc=medium_risk_ttc))
    assessor.high_risk_ttc = high_risk_ttc
    assessor.medium_risk_ttc = medium_risk_ttc
    st.session_state.assessor = assessor
    
    visualizer = st.session_state.get("visualizer", Visualizer())
    st.session_state.visualizer = visualizer
    
    is_live = input_source in ["Live Webcam (0)", "YouTube Live Stream"]
    if is_live:
        video_stream = get_or_create_stream_buffer(video_path, resize_dims=(640, 360))
        native_fps = 30.0
        total_frames = None
    else:
        # File stream playback: reuse open stream so pause does not restart playback from frame 0
        if (
            "file_stream" not in st.session_state
            or st.session_state.file_stream is None
            or st.session_state.get("current_file_path") != video_path
            or not st.session_state.file_stream.isOpened()
        ):
            if "file_stream" in st.session_state and st.session_state.file_stream is not None:
                try:
                    st.session_state.file_stream.release()
                except Exception:
                    pass
            st.session_state.file_stream = cv2.VideoCapture(video_path)
            st.session_state.current_file_path = video_path
        video_stream = st.session_state.file_stream

        # Read native video FPS and frame count
        raw_fps = video_stream.get(cv2.CAP_PROP_FPS)
        if raw_fps is None or raw_fps <= 0 or np.isnan(raw_fps) or raw_fps > 120:
            native_fps = 25.0
        else:
            native_fps = float(raw_fps)
        raw_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = raw_frames if raw_frames > 0 else None

    target_frame_interval = 1.0 / max(1, target_fps)
    cap = video_stream
        
    if not video_stream.isOpened():
        st.error(f"Failed to open video source: {input_source}")
        st.session_state.is_running = False
        st.stop()

    # --- UI LAYOUT ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        status_placeholder = st.empty()
        st_frame = st.empty()
        frame_placeholder = st_frame  # Alias for backward compatibility
        st.subheader("Analytics Summary")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            frame_metric = st.empty()
        with metric_col2:
            hazard_metric = st.empty()
        with metric_col3:
            fps_metric = st.empty()
        
    with col2:
        st.markdown("### Interactive Logs")
        log_expander = st.expander("Pipeline Logs & Telemetry", expanded=True)
        log_container = log_expander.empty()
        
    # State tracking
    if 'loop_logs' not in st.session_state:
        st.session_state.loop_logs = []
    if 'total_high_risks' not in st.session_state:
        st.session_state.total_high_risks = set()
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'prev_time' not in st.session_state:
        st.session_state.prev_time = time.time()
    if 'smoothed_fps' not in st.session_state:
        st.session_state.smoothed_fps = 0.0

    frame_count = st.session_state.frame_count
    prev_time = st.session_state.prev_time
    smoothed_fps = st.session_state.smoothed_fps
    total_high_risks = st.session_state.total_high_risks

    # --- PAUSED STATE HANDLING ---
    # When paused: hold the last captured frame on screen without re-initializing or restarting the stream
    if st.session_state.paused:
        if st.session_state.last_frame is not None:
            st_frame.image(st.session_state.last_frame, channels="RGB")
        status_placeholder.warning("⏸️ Playback Paused: Stream frozen on current frame. Click '▶️ Resume Stream' to continue.")
        frame_text = f"#{frame_count}" if total_frames is None else f"#{frame_count} / {total_frames}"
        frame_metric.metric("Current Frame", f"{frame_text} (Paused)")
        hazard_metric.metric("High-Risk Objects", len(total_high_risks))
        fps_metric.metric("Processing FPS", "PAUSED")
        if len(st.session_state.loop_logs) > 0:
            display_logs = "\n".join(st.session_state.loop_logs[-15:])
            log_container.code(display_logs, language="text", wrap_lines=True)
    else:
        # --- ACTIVE PROCESSING LOOP ---
        status_placeholder.empty()
        last_log_update = 0.0
        last_metric_update = 0.0
        while st.session_state.get("is_running", False) and video_stream.isOpened():
            # Check if pause was requested during execution
            if st.session_state.get("paused", False):
                break

            frame_start_time = time.time()

            # 1. Thread-safe frame read
            if is_live:
                # In live mode: wait for the camera's next fresh hardware frame (up to 80ms)
                ret, frame = video_stream.read(wait_for_new=True, timeout=0.08)
            else:
                ret, frame = video_stream.read()

            if not ret or frame is None:
                # End of file or dead stream
                break

            # 2. Frame normalization: resize incoming frame to 640x360
            if frame.shape[1] != 640 or frame.shape[0] != 360:
                frame = cv2.resize(frame, (640, 360))

            # FPS Calculation & Smoothing
            curr_time = time.time()
            instant_fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            smoothed_fps = instant_fps if frame_count == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            fps = smoothed_fps
            st.session_state.prev_time = prev_time
            st.session_state.smoothed_fps = smoothed_fps

            # 3. Perception: YOLOv8 inference with optional frame alternating for CPU speedup
            run_yolo = True
            if inference_mode == "Every 2nd Frame (High Speed 40+ FPS)" and frame_count % 2 == 1:
                run_yolo = False

            if run_yolo:
                try:
                    detections = detector.detect(frame, imgsz=selected_imgsz)
                except TypeError:
                    detections = detector.detect(frame)
                st.session_state.last_detections = detections
            else:
                # Passing empty detections lets the tracker's dead reckoning advance tracked objects smoothly
                detections = []

            # 4. Tracking: Multi-object centroid tracking & velocity estimation
            tracked_objects = tracker.update(detections, frame)

            # 5. Prediction: Trajectory forecasting & Time-to-Collision (TTC)
            predicted_objects = forecaster.forecast(tracked_objects)

            # 6. Risk Assessment: Rule-based hazard evaluation
            risk_alerts = assessor.assess(predicted_objects)

            # Telemetry logging & hazard tracking
            telemetry_msg = f"⏱️ [Frame #{frame_count + 1}] FPS: {fps:.1f} | Detections: {len(detections)} | Tracked: {len(tracked_objects)}"
            st.session_state.loop_logs.append(telemetry_msg)

            has_high_risk = False
            for obj_id, risk_level in risk_alerts.items():
                if risk_level == 'High':
                    total_high_risks.add(obj_id)
                    has_high_risk = True
                    log_entry = f"🔴 [Frame #{frame_count + 1}] HIGH RISK: Object #{obj_id} approaching fast!"
                    if log_entry not in st.session_state.loop_logs:
                        st.session_state.loop_logs.append(log_entry)
                elif risk_level == 'Medium':
                    log_entry = f"🟡 [Frame #{frame_count + 1}] CAUTION: Object #{obj_id} shows medium risk."
                    if len(st.session_state.loop_logs) == 0 or st.session_state.loop_logs[-1] != log_entry:
                        st.session_state.loop_logs.append(log_entry)

            # Keep logs trimmed in memory to prevent DOM degradation
            if len(st.session_state.loop_logs) > 100:
                st.session_state.loop_logs = st.session_state.loop_logs[-50:]

            # 7. Visualization: Draw predictive bounding boxes, trails, and trajectory vectors + HUD overlay
            annotated_frame = visualizer.draw_annotations(frame, predicted_objects, risk_alerts, fps=fps, frame_idx=frame_count + 1)

            # Convert BGR (OpenCV) to RGB (Streamlit)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame = annotated_frame_rgb
            st.session_state.last_annotated_frame = annotated_frame_rgb

            # Render updated visual frame - ONLY image updates every cycle to avoid WebSocket congestion
            st_frame.image(annotated_frame_rgb, channels="RGB")

            frame_count += 1
            st.session_state.frame_count = frame_count

            now = time.time()

            # Rate-limit metric updates to 4 times per second (every 0.25s) to eliminate React DOM reflow lag
            if now - last_metric_update >= 0.25:
                last_metric_update = now
                frame_text = f"#{frame_count}" if total_frames is None else f"#{frame_count} / {total_frames}"
                frame_metric.metric("Current Frame", frame_text)
                hazard_metric.metric("High-Risk Objects", len(total_high_risks))
                fps_metric.metric("Processing FPS", f"{fps:.1f}")

            # Decoupled log rendering: rate-limited by time (0.5s) to eliminate DOM stutter & frame freezing
            if now - last_log_update >= 0.5 or has_high_risk:
                last_log_update = now
                display_logs = "\n".join(st.session_state.loop_logs[-15:])
                log_container.code(display_logs, language="text", wrap_lines=True)

            # Universal Frame Pacing: ensures playback strictly matches target_fps across all sources
            frame_elapsed = time.time() - frame_start_time
            sleep_time = target_frame_interval - frame_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                time.sleep(0.001)

        # Loop cleanup on stream termination
        if not st.session_state.paused:
            if not is_live:
                if "file_stream" in st.session_state and st.session_state.file_stream is not None:
                    try:
                        st.session_state.file_stream.release()
                    except Exception:
                        pass
                    st.session_state.file_stream = None
                st.session_state.is_running = False
                st.session_state.paused = False
                st.success("Analysis Complete!")
            elif not video_stream.isOpened():
                release_stream_buffer()
                st.session_state.is_running = False
                st.session_state.paused = False
                st.error("Live video stream disconnected or ended.")
else:
    st.info("👈 Select your video source in the sidebar and click **🚀 Run Analysis** to start proactive hazard detection.")
