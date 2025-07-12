import os
import traceback
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, make_response
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops, ExifTags
import cv2
import ffmpeg
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import math
import shutil

# PySceneDetect Imports
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector

# --- Configuration & Setup ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads/')
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_EXTENSIONS_VID = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'mpg', 'mpeg', 'wmv'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024 # Increased for larger video files
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = 'super-secret-key'

# --- Helper Functions (No changes here) ---
def allowed_file(filename, extensions): return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions
def get_image_metadata(filepath):
    try:
        img = Image.open(filepath); exif_data = img._getexif()
        if not exif_data: return {"Status": "No EXIF metadata found."}
        metadata = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items()}
        for key, value in metadata.items():
            if isinstance(value, bytes):
                try: metadata[key] = value.decode('utf-8', errors='ignore')
                except: metadata[key] = str(value)
        return metadata
    except Exception as e: return {"Error": f"Could not read metadata: {e}"}
def perform_ela(filepath, quality=90):
    try:
        original = Image.open(filepath).convert('RGB')
        temp_file = BytesIO(); original.save(temp_file, 'JPEG', quality=quality); temp_file.seek(0)
        resaved = Image.open(temp_file).convert('RGB'); ela_image = ImageChops.difference(original, resaved)
        extrema = ela_image.getextrema(); max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0: max_diff = 1; scale = 255.0 / max_diff
        ela_image = ImageChops.multiply(ela_image, Image.new('RGB', ela_image.size, (int(scale),) * 3))
        buffer = BytesIO(); ela_image.save(buffer, format="PNG"); return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e: print(f"ELA Error: {e}"); return None
def plot_image_histogram(filepath):
    try:
        img = Image.open(filepath).convert('RGB')
        r, g, b = img.split(); hist_r = r.histogram(); hist_g = g.histogram(); hist_b = b.histogram()
        plt.switch_backend('Agg'); fig = plt.figure(figsize=(10, 4))
        plt.plot(hist_r, color='red', alpha=0.7); plt.plot(hist_g, color='green', alpha=0.7); plt.plot(hist_b, color='blue', alpha=0.7)
        plt.title('Color Channel Histogram'); plt.xlabel('Pixel Intensity (0-255)'); plt.ylabel('Pixel Count')
        plt.grid(True); plt.legend(['Red', 'Green', 'Blue']); plt.xlim(0, 255)
        buffer = BytesIO(); plt.savefig(buffer, format='png', bbox_inches='tight'); plt.close(fig); buffer.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e: print(f"Histogram Error: {e}"); return None
def get_video_metadata(filepath):
    try: return ffmpeg.probe(filepath)['format']
    except ffmpeg.Error as e: return {"Error": f"Could not probe video: {e.stderr}"}
def extract_video_frames(filepath, num_frames=5):
    try:
        cap = cv2.VideoCapture(filepath); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int); extracted_frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ret, frame = cap.read()
            if ret: _, buffer = cv2.imencode('.jpg', frame); extracted_frames.append(f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}")
        cap.release(); return extracted_frames
    except Exception as e: print(f"Frame Extraction Error: {e}"); return []
def get_video_fps(filepath):
    try:
        probe = ffmpeg.probe(filepath)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream and 'avg_frame_rate' in video_stream and '/' in video_stream['avg_frame_rate']:
            num, den = video_stream['avg_frame_rate'].split('/'); return float(num) / float(den)
    except Exception: return 30.0
def analyze_audio_anomalies(y, sr, fps, stdev_threshold=3.0):
    anomalies = { "amplitude": [], "frequency": [] };
    if y.size == 0: return anomalies
    rms = librosa.feature.rms(y=y)[0]; rms_diff = np.diff(rms)
    if rms_diff.size > 0:
        rms_threshold = np.mean(rms_diff) + (stdev_threshold * np.std(rms_diff))
        rms_anomaly_indices = np.where(rms_diff > rms_threshold)[0]
        anomaly_times = librosa.frames_to_time(rms_anomaly_indices, sr=sr); anomalies["amplitude"] = [int(t * fps) for t in anomaly_times]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]; centroid_diff = np.diff(centroid)
    if centroid_diff.size > 0:
        centroid_threshold = np.mean(centroid_diff) + (stdev_threshold * np.std(centroid_diff))
        centroid_anomaly_indices = np.where(centroid_diff > centroid_threshold)[0]
        anomaly_times = librosa.frames_to_time(centroid_anomaly_indices, sr=sr); anomalies["frequency"] = [int(t * fps) for t in anomaly_times]
    return anomalies
def plot_audio_waveform(y, sr, fps):
    try:
        plt.switch_backend('Agg'); fig, ax = plt.subplots(figsize=(10, 3)); librosa.display.waveshow(y, sr=sr, ax=ax, color='blue')
        ax.set_title("Audio Waveform (Amplitude vs. Time)"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
        ax.grid(True); buffer = BytesIO(); plt.savefig(buffer, format='png', bbox_inches='tight'); plt.close(fig); buffer.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e: print(f"Waveform Plot Error: {e}"); return None
def plot_audio_spectrogram(y, sr):
    try:
        if y.size == 0: return None
        plt.switch_backend('Agg'); fig, ax = plt.subplots(figsize=(10, 4));
        D = librosa.stft(y); S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB'); ax.set_title('Audio Spectrogram (Frequency vs. Time)')
        buffer = BytesIO(); plt.savefig(buffer, format='png', bbox_inches='tight'); plt.close(fig); buffer.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e: print(f"Spectrogram Plot Error: {e}"); return None

### --- NEW FUNCTION: OPTICAL FLOW SCENE ANALYSIS --- ###
def analyze_scenes_with_optical_flow(video_path):
    """
    Analyzes a video for scene changes using a combination of optical flow and
    pixel difference to be more robust against camera motion.
    NOTE: This method is significantly slower than PySceneDetect.
    """
    report = {"changes": []}
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"Error": "Could not open video file."}

        # --- TUNABLE PARAMETERS ---
        # How much pixel difference is needed to be considered a potential cut.
        # Lower this if cuts are missed. Higher if there are too many detections.
        PIXEL_DIFF_THRESHOLD = 300.0

        # How much average motion is allowed before we assume it's a pan/zoom.
        # Higher this value if slow pans are still detected as cuts.
        # Lower this if actual cuts in high-motion scenes are missed.
        FLOW_MAG_THRESHOLD = 1.0

        ret, prev_frame = cap.read()
        if not ret:
            return {"Error": "Could not read the first frame."}
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_num = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Calculate Optical Flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitude = np.mean(magnitude)

            # 2. Calculate Pixel Difference (Mean Squared Error)
            pixel_diff = np.mean((prev_gray.astype("float") - current_gray.astype("float")) ** 2)

            # 3. Make a decision
            # A real cut has high pixel difference BUT low, non-uniform motion.
            if pixel_diff > PIXEL_DIFF_THRESHOLD and flow_magnitude < FLOW_MAG_THRESHOLD:
                report["changes"].append(frame_num)

            # Update for next iteration
            prev_gray = current_gray
            frame_num += 1

        cap.release()
        return report

    except Exception as e:
        print(f"Optical Flow Analysis Error: {e}")
        traceback.print_exc()
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return {"changes": [], "Error": str(e)}

# --- This function remains for comparison or faster analysis if needed ---
def expert_scene_analysis_adaptive(video_path):
    report = {"changes": []}
    try:
        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(AdaptiveDetector(adaptive_threshold=3.0))
        sm.detect_scenes(frame_source=video)
        scene_list = sm.get_scene_list()
        if len(scene_list) > 0:
            report["changes"] = sorted(list(set([scene[1].get_frames() for scene in scene_list])))
        return report
    except Exception as e:
        print(f"PySceneDetect Error: {e}"); traceback.print_exc()
        return report

# --- Main App Routes ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files: return "No file part", 400
    file = request.files['file']
    if file.filename == '': return "No selected file", 400
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_VID):
        filename = secure_filename(file.filename)
        if not filename: return "Invalid filename. Please rename the file and try again.", 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename); file.save(filepath)
        
        print("Starting video analysis...")
        fps = get_video_fps(filepath)
        metadata = get_video_metadata(filepath)
        frames_preview = extract_video_frames(filepath)
        
        # --- Using the new, more accurate (but slower) optical flow method ---
        print("Performing Optical Flow scene analysis...")
        scene_change_report = analyze_scenes_with_optical_flow(filepath)
        print(f"Optical Flow analysis complete. Report: {scene_change_report}")

        # --- Audio Analysis ---
        try:
            y, sr = librosa.load(filepath, sr=None)
        except Exception as e:
            y, sr = np.array([]), 0
            print(f"Could not load audio. Error: {e}")

        waveform_plot = plot_audio_waveform(y, sr, fps)
        spectrogram_plot = plot_audio_spectrogram(y, sr)
        audio_anomalies = analyze_audio_anomalies(y, sr, fps)
        total_frames_val = int(float(metadata.get('duration', 0.0)) * fps) if metadata and 'duration' in metadata else 0
        
        return render_template('results_video.html', filename=filename, metadata=metadata, frames=frames_preview, scene_report=scene_change_report, waveform_plot=waveform_plot, spectrogram_plot=spectrogram_plot, audio_anomalies=audio_anomalies, total_frames=total_frames_val)
    return "Invalid file type. Please upload a valid video.", 400

# (The rest of the file: /upload-image, /upload-multiple-images, etc., remains exactly the same)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files: return "No file part", 400
    file = request.files['file']
    if file.filename == '': return "No selected file", 400
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_IMG):
        filename = secure_filename(file.filename)
        if not filename: return "Invalid filename. Please rename the file and try again.", 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename); file.save(filepath)
        metadata = get_image_metadata(filepath); ela_image_b64 = perform_ela(filepath); histogram_plot = plot_image_histogram(filepath)
        return render_template('results_image.html', filename=filename, metadata=metadata, ela_image_b64=ela_image_b64, histogram_plot=histogram_plot)
    return "Invalid file type. Please upload a valid image.", 400
@app.route('/upload-multiple-images', methods=['POST'])
def upload_multiple_images():
    uploaded_files = request.files.getlist('files')
    if not uploaded_files or uploaded_files[0].filename == '': return "No selected files", 400
    results = []
    for file in uploaded_files:
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_IMG):
            filename = secure_filename(file.filename)
            if not filename: continue
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename); file.save(filepath)
            metadata = get_image_metadata(filepath); ela_image_b64 = perform_ela(filepath); histogram_plot = plot_image_histogram(filepath)
            results.append({'filename': filename, 'metadata': metadata, 'ela_image_b64': ela_image_b64, 'histogram_plot': histogram_plot})
    if not results: return "No valid image files were uploaded.", 400
    return render_template('results_multiple.html', results=results)
def deconstruct_video_core(filepath, filename):
    base_name = os.path.splitext(filename)[0]; frames_dir_name = f"{base_name}_frames"
    frames_path = os.path.join(app.config['UPLOAD_FOLDER'], frames_dir_name)
    if os.path.exists(frames_path): shutil.rmtree(frames_path)
    os.makedirs(frames_path)
    cap = cv2.VideoCapture(filepath); count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_filename = f"frame_{count:05d}.jpg"; cv2.imwrite(os.path.join(frames_path, frame_filename), frame); count += 1
    cap.release(); return frames_dir_name
@app.route('/extract-only', methods=['POST'])
def extract_only():
    if 'file' not in request.files: return "No file part", 400
    file = request.files['file']
    if file.filename == '': return "No selected file", 400
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_VID):
        filename = secure_filename(file.filename)
        if not filename: return "Invalid filename. Please rename the file and try again.", 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath); frames_dir_name = deconstruct_video_core(filepath, filename)
        return redirect(url_for('view_frames', frames_dir=frames_dir_name))
    return "Invalid file type. Please upload a valid video.", 400
@app.route('/deconstruct/<filename>', methods=['POST'])
def deconstruct_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path): return "Video file not found.", 404
    frames_dir_name = deconstruct_video_core(video_path, filename)
    return redirect(url_for('view_frames', frames_dir=frames_dir_name))
@app.route('/view-frames/<frames_dir>')
def view_frames(frames_dir):
    frames_path = os.path.join(app.config['UPLOAD_FOLDER'], frames_dir)
    if not os.path.isdir(frames_path): return "Frames directory not found.", 404
    page = request.args.get('page', 1, type=int); per_page = 50
    try: all_frames = sorted(os.listdir(frames_path))
    except FileNotFoundError: return "Frames directory not found.", 404
    total_frames = len(all_frames); total_pages = math.ceil(total_frames / per_page)
    start_index = (page - 1) * per_page; end_index = start_index + per_page
    frames_to_show = all_frames[start_index:end_index]
    frame_paths = [os.path.join(frames_dir, f).replace('\\', '/') for f in frames_to_show]
    return render_template('view_frames.html', frames=frame_paths, frames_dir=frames_dir, current_page=page, total_pages=total_pages)
@app.route('/download-zip/<frames_dir>')
def download_zip(frames_dir):
    directory_path = os.path.join(app.config['UPLOAD_FOLDER'], frames_dir)
    if not os.path.isdir(directory_path): return "Directory not found.", 404
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{frames_dir}.zip")
    try:
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', directory_path)
        response = make_response(send_from_directory(app.config['UPLOAD_FOLDER'], f"{frames_dir}.zip", as_attachment=True))
        response.headers["Content-Disposition"] = f"attachment; filename={frames_dir}.zip"
        @response.call_on_close
        def atexit_remove_zip():
            if os.path.exists(zip_path): os.remove(zip_path)
        return response
    except Exception as e: print(f"ZIP creation error: {e}"); return "Could not create ZIP file.", 500
@app.route('/view-image/<path:image_path>')
def view_image(image_path):
    return render_template('view_image.html', image_path=image_path)
@app.route('/uploads/<path:path>')
def send_upload(path):
    return send_from_directory(app.config['UPLOAD_FOLDER'], path)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='127.0.0.1', port=5000)