import cv2
import time
import numpy as np
import os
from datetime import datetime
from collections import deque
from flask import Flask, render_template, send_from_directory, jsonify
from flask_sock import Sock
from turbojpeg import TurboJPEG
from zoneinfo import ZoneInfo
import threading
import pyaudio
import queue
from ultralytics import YOLO

# ==========================================
# AUDIO SETUP & THREADING (SMOOTH AUDIO FIX)
# ==========================================
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 44100
# Increased chunk size to stop network stuttering
AUDIO_CHUNK = 4096  

audio = pyaudio.PyAudio()

def find_microphone_index(target_name="Lenovo"):
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0 and target_name in device_info.get('name', ''):
            print(f"[+] Found {target_name} Audio at Index: {i}")
            return i
    print(f"[-] CRITICAL: Could not find a microphone named {target_name}.")
    return None

MIC_INDEX = find_microphone_index("Lenovo")

if MIC_INDEX is not None:
    audio_mic = audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE,
                           input=True, input_device_index=MIC_INDEX, frames_per_buffer=AUDIO_CHUNK)
else:
    print("[!] Falling back to system default microphone...")
    audio_mic = audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE,
                           input=True, frames_per_buffer=AUDIO_CHUNK)

# Create a queue to hold audio chunks so the microphone never waits for the network
audio_queue = queue.Queue(maxsize=10)

def audio_capture_worker():
    print("[+] Audio capture thread started.")
    while True:
        try:
            data = audio_mic.read(AUDIO_CHUNK, exception_on_overflow=False)
            if not audio_queue.full():
                audio_queue.put(data)
        except Exception as e:
            print(f"[-] Audio hardware error: {e}")
            break

# Start the audio gathering on a separate background thread
audio_thread = threading.Thread(target=audio_capture_worker, daemon=True)
audio_thread.start()

# ==========================================
# FLASK & CAMERA SETUP
# ==========================================
RECORDING_DIR = "recordings"
os.makedirs(RECORDING_DIR, exist_ok=True)

app = Flask(__name__)
sock = Sock(app)
jpeg = TurboJPEG()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

print("Waking up camera buffer...")
for _ in range(10):
    cap.read()
    time.sleep(0.05)

print("Setting Camera to Manual Mode and applying baseline settings...")
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  
current_exposure = 157  
current_gain = 10       
frame_counter = 0       

cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
cap.set(cv2.CAP_PROP_GAIN, current_gain)

# ==========================================
# YOLOv8 NANO SETUP
# ==========================================
print("[*] Loading YOLOv8 Nano model...")
model = YOLO("yolov8n.pt") 
PERSON_CLASS_ID = 0

# ==========================================
# RECORDING BUFFER SETTINGS
# ==========================================
FPS_ESTIMATE = 30
PRE_RECORD_SECONDS = 3  
COOLDOWN_SECONDS = 4    

frame_buffer = deque(maxlen=PRE_RECORD_SECONDS * FPS_ESTIMATE)
is_recording = False
video_writer = None
frames_since_last_detection = 0
last_known_detections = []
latest_jpeg = None

# ==========================================
# MAIN CAMERA LOOP
# ==========================================
def camera_worker():
  global current_exposure, current_gain, frame_counter, latest_jpeg
  global is_recording, video_writer, frames_since_last_detection, frame_buffer, last_known_detections
  
  print("[+] WEBSOCKET UPGRADE SUCCESSFUL! Browser connected.")
  
  if not cap.isOpened():
      print("[-] CRITICAL: Camera is physically disconnected.")
      return

  while True:
      try:
          ret, frame = cap.read()
          if not ret:
              break 
          
          frame_counter += 1

          # --- 1. HARDWARE ADAPTIVE LOOP (Must read RAW light) ---
          if frame_counter % 6 == 0:  
              gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
              blown_out_pixels = np.sum(gray_raw > 240) 
              glare_ratio = blown_out_pixels / gray_raw.size 
              avg_brightness = np.mean(gray_raw) 
              
              if glare_ratio > 0.3 or avg_brightness > 450:
                  if current_gain > 0:
                      current_gain = max(0, current_gain - 20)       
                      cap.set(cv2.CAP_PROP_GAIN, current_gain)
                  elif current_exposure > 157:
                      current_exposure = max(157, current_exposure - 400)  
                      cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
                      
              elif avg_brightness < 400 and glare_ratio < 0.2:
                  exposure_step = 600 if avg_brightness < 50 else 200
                  if current_exposure < 10000:  
                      current_exposure = min(10000, current_exposure + exposure_step)
                      cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
                  elif current_gain < 250:      
                      current_gain = min(250, current_gain + 15)       
                      cap.set(cv2.CAP_PROP_GAIN, current_gain)

          # --- 2. SOFTWARE CONTRAST (After hardware adapts) ---
          contrast_boost = 0.8  
          haze_reduction = -60  
          frame = cv2.convertScaleAbs(frame, alpha=contrast_boost, beta=haze_reduction)

          clean_frame = frame.copy()
          frame_buffer.append(clean_frame)

          # --- 3. YOLO AI DETECTION ---
          (h, w) = frame.shape[:2]
          person_detected_this_cycle = False 
          
          if frame_counter % 5 == 0:
              ai_frame = cv2.GaussianBlur(frame, (5, 5), 0)
              results = model(ai_frame, classes=[PERSON_CLASS_ID], conf=0.3, verbose=False, device='cpu')
              last_known_detections = [] 
              
              for result in results:
                  boxes = result.boxes
                  for box in boxes:
                      x1, y1, x2, y2 = box.xyxy[0]
                      startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)
                      confidence = float(box.conf[0])
                      
                      # Ghost Filtering
                      box_height = endY - startY
                      if box_height > (h * 0.15) and box_height < (h * 0.95):
                          person_detected_this_cycle = True
                          last_known_detections.append(((startX, startY, endX, endY), confidence))

          for (box, confidence) in last_known_detections:
              (startX, startY, endX, endY) = box
              label = f"Person: {confidence * 100:.1f}%"
              cv2.rectangle(frame, (startX, startY), (endX, endY), (16, 185, 129), 2)
              y = startY - 15 if startY - 15 > 15 else startY + 15
              cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 185, 129), 2)
              person_detected_this_cycle = True

          # --- 4. RECORDING LOGIC ---
          if person_detected_this_cycle:
              frames_since_last_detection = 0
              if not is_recording:
                  is_recording = True
                  timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d_%H-%M-%S")
                  filename = os.path.join(RECORDING_DIR, f"Detection_{timestamp}.webm")
                  fourcc = cv2.VideoWriter_fourcc(*'VP80')
                  video_writer = cv2.VideoWriter(filename, fourcc, FPS_ESTIMATE, (w, h))
                  print(f"[*] Recording live to {filename}...")
                  for buffered_frame in frame_buffer:   
                      video_writer.write(buffered_frame)
          else:
              if is_recording:
                  frames_since_last_detection += 1

          if is_recording:
              video_writer.write(clean_frame)
              if frames_since_last_detection > (COOLDOWN_SECONDS * FPS_ESTIMATE):
                  print(f"[*] Person left. Saving {filename}.")
                  is_recording = False
                  video_writer.release()
                  video_writer = None

          # Stream output
          latest_jpeg = jpeg.encode(frame, quality=70)
          time.sleep(0.005)

      except Exception as e:
            print(f"[-] ERROR Camera loop: {e}")
            break
      
camera_thread = threading.Thread(target=camera_worker, daemon=True)
camera_thread.start()

# ==========================================
# FLASK ROUTING
# ==========================================
@app.route('/')
def index():
    return render_template('main.html')

@sock.route('/stream')
def stream(ws):
  while True:
        try:
            if latest_jpeg is not None:
                ws.send(latest_jpeg)
            time.sleep(0.05) 
        except Exception:
            break
        
@sock.route('/audio_stream')
def audio_websocket(ws):
    while True:
        try:
            # Grab audio from the background thread queue instead of reading mic directly
            data = audio_queue.get()
            ws.send(data)
        except Exception:
            break

@app.route('/api/videos')
def list_videos():
    files = os.listdir(RECORDING_DIR)
    videos = [f for f in files if f.endswith('.webm')]
    videos.sort(reverse=True) 
    return jsonify({"videos": videos})

@app.route('/recordings/<filename>')
def serve_video(filename):
    return send_from_directory(RECORDING_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)