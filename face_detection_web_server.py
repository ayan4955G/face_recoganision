from flask import Flask, Response, render_template_string, jsonify, request
import time
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from deepface import DeepFace
import os
import json

# ---------- CONFIG ----------
RTSP_URL = "rtsp://admin:L2E9B7FC@10.47.29.210:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
# Use RTSP_URL or 0 for webcam
VIDEO_SOURCE = 0  # change to RTSP_URL to use camera stream

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_FACE_CONF = 0.95
COSINE_THRESHOLD = 0.45
SVM_PROB_THRESHOLD = 0.50
EMO_EVERY_N = 10

people = set()
# Better approach - automatically closes the file
with open("students.json", "r") as file:
    students_data = json.load(file)

# ---------- LOAD MODELS ----------
print("Loading models...")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
mtcnn = MTCNN(keep_all=False, device=DEVICE if DEVICE == 'cuda' else 'cpu')

base_dir = os.path.dirname(os.path.realpath(__file__))
emb_file = os.path.join(base_dir, "embeddings.npy")
labels_file = os.path.join(base_dir, "embedding_labels.npy")
label_map_file = os.path.join(base_dir, "label_map.npy")
svm_path = os.path.join(base_dir, "svm_model.joblib")
oneclass_path = os.path.join(base_dir, "oneclass_model.joblib")

if not os.path.exists(emb_file) or not os.path.exists(labels_file) or not os.path.exists(label_map_file):
    raise FileNotFoundError("embeddings.npy, embedding_labels.npy or label_map.npy missing in model folder")

embeddings_db = np.load(emb_file)
labels_db = np.load(labels_file)
label_map = np.load(label_map_file, allow_pickle=True).item()
embeddings_db = normalize(embeddings_db)
unique_labels = np.unique(labels_db)
if len(unique_labels) == 0:
    raise RuntimeError("No labels found in label DB.")
class_means_matrix = np.vstack([np.mean(embeddings_db[labels_db == u], axis=0)
                                for u in unique_labels])
class_order = list(unique_labels)

model = None
model_type = "none"
if os.path.exists(svm_path):
    model = joblib.load(svm_path)
    model_type = "svm"
    print("Loaded multi-class SVM:", svm_path)
elif os.path.exists(oneclass_path):
    model = joblib.load(oneclass_path)
    model_type = "oneclass"
    print("Loaded OneClassSVM:", oneclass_path)
else:
    print("No classifier file found. Will use cosine-only matching.")

print("Models loaded.")

# caching emotion/name per bounding box
class FaceTrack:
    def __init__(self, name):
        self.name = name
        self.last_emo_frame = -999
        self.emotion = "N/A"

tracks = {}

# open capture
# cap = cv2.VideoCapture(RTSP_URL)
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError("Cannot open video stream")

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facial Attendance System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .container {

            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }



        .main-content {
            display: grid;
            grid-template-columns: 1fr 1.5fr;
            gap: 30px;
            padding: 30px;
            height: 100vh;
            width: 100vw;
        }

        .video-section {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .video-container {
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        #videoFeed {
            width: 100%;
            height: auto;
            display: block;
        }

        .video-placeholder {
            width: 100%;
            aspect-ratio: 4/3;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 18px;
        }

        .controls {
            display: flex;
            gap: 10px;
        }

        .btn {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #f1f3f5;
            color: #495057;
        }

        .btn-secondary:hover {
            background: #e9ecef;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }

        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-number {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
        }

        .stat-card.present .stat-number { color: #28a745; }
        .stat-card.absent .stat-number { color: #dc3545; }
        .stat-card.late .stat-number { color: #ffc107; }

        .table-section {
            display: flex;
            flex-direction: column;
        }

        .table-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .table-header h2 {
            font-size: 20px;
            color: #212529;
        }

        .search-box {
            padding: 8px 15px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 14px;
            width: 250px;
        }

        .search-box:focus {
            outline: none;
            border-color: #667eea;
        }

        .table-container {
            overflow-y: auto;
            max-height: 1000px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
    
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        thead {
            background: #f8f9fa;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        th {
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
            font-size: 13px;
            text-transform: uppercase;
        }

        td {
            padding: 15px;
            border-bottom: 1px solid #f1f3f5;
            font-size: 14px;
        }

        tbody tr {
            transition: background-color 0.2s ease;
        }

        tbody tr:hover {
            background-color: #f8f9fa;
        }

        .status-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-present {
            background: #d4edda;
            color: #155724;
        }

        .status-absent {
            background: #f8d7da;
            color: #721c24;
        }

        .status-late {
            background: #fff3cd;
            color: #856404;
        }

        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">

        <div class="main-content">
            <div class="video-section">
                <div class="video-container">
                    <div class="video-placeholder">
                        <img id="stream" src="{{ url_for('video_feed') }}" style="max-width:100%; height:100%"/>
                    </div>
                </div>

   

                <div class="stats">
                    <div class="stat-card present">
                        <div class="stat-number" id="presentCount">0</div>
                        <div class="stat-label">Present</div>
                    </div>
                    <div class="stat-card absent">
                        <div class="stat-number" id="absentCount">0</div>
                        <div class="stat-label">Absent</div>
                    </div>
                    <div class="stat-card late">
                        <div class="stat-number" id="lateCount">0</div>
                        <div class="stat-label">Late</div>
                    </div>
                </div>
            </div>

            <div class="table-section">
                <div class="table-header">
                    <h2>Student Attendance</h2>
                    <input type="text" class="search-box" id="searchBox" placeholder="Search students..." onkeyup="filterTable()">
                </div>

                <div class="table-container">
                    <table id="attendanceTable">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Roll No</th>
                                <th>Standard</th>
                                <th>Division</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

   <script>
    let students = [];

    async function fetchAttendanceData() {
        try {
            const response = await fetch('/get_attendance_data');
            const data = await response.json();
            students = data;
            renderTable(students);
        } catch (error) {
            console.error('Error fetching attendance data:', error);
        }
    }

    function renderTable(data = students) {
        const tbody = document.getElementById('tableBody');
        tbody.innerHTML = '';

        // Sort students: present first, then late, then absent
        const sortedData = [...data].sort((a, b) => {
            const statusOrder = { 'present': 0, 'late': 1, 'absent': 2 };
            return statusOrder[a.status] - statusOrder[b.status];
        });

        sortedData.forEach(student => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${student.name}</td>
                <td>${student.rollNo}</td>
                <td>${student.standard}</td>
                <td>${student.division}</td>
                <td><span class="status-badge status-${student.status}">${student.status}</span></td>
            `;
            tbody.appendChild(row);
        });

        updateStats(sortedData);
    }

    function updateStats(data = students) {
        const present = data.filter(s => s.status === 'present').length;
        const absent = data.filter(s => s.status === 'absent').length;
        const late = data.filter(s => s.status === 'late').length;

        document.getElementById('presentCount').textContent = present;
        document.getElementById('absentCount').textContent = absent;
        document.getElementById('lateCount').textContent = late;
    }

    function filterTable() {
        const searchTerm = document.getElementById('searchBox').value.toLowerCase();
        const filtered = students.filter(student => 
            student.name.toLowerCase().includes(searchTerm) ||
            student.rollNo.toLowerCase().includes(searchTerm) ||
            student.standard.toLowerCase().includes(searchTerm) ||
            student.division.toLowerCase().includes(searchTerm)
        );
        renderTable(filtered);
    }

    function startCamera() {
        document.getElementById('stream').src = '/video_feed';
    }

    function stopCamera() {
        document.getElementById('stream').src = '';
    }

    // Initial load
    fetchAttendanceData();

    // Refresh attendance data every 3 seconds
    setInterval(fetchAttendanceData, 3000);
</script>
</body>
</html>
"""

def gen_frames():
    frame_id = 0
    t0 = time.time()
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        frame_id += 1
        h, w = frame.shape[:2]

        # detect faces
        boxes, probs = mtcnn.detect(frame)

        if boxes is not None:
            for i, box in enumerate(boxes):
                prob = probs[i] if probs is not None else 1.0
                if prob < MIN_FACE_CONF:
                    continue

                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # embedding
                face_resized = cv2.resize(face, (160, 160))
                rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                t = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                t = (t - 127.5) / 128.0
                with torch.no_grad():
                    emb = resnet(t).cpu().numpy()
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

                # identification logic (same as your script)
                name = "Unknown"
                is_known = False

                if model_type == "svm":
                    try:
                        probs_out = model.predict_proba(emb)[0]
                        pred_idx = int(np.argmax(probs_out))
                        svc_prob = probs_out[pred_idx]
                        pred_label = model.classes_[pred_idx]

                        sims = cosine_similarity(emb.reshape(1, -1), class_means_matrix)[0]
                        best_idx = int(np.argmax(sims))
                        cos_dist = 1 - sims[best_idx]

                        is_known = (svc_prob >= SVM_PROB_THRESHOLD) and (cos_dist <= COSINE_THRESHOLD)
                        if is_known:
                            name = label_map[int(pred_label)]
                            people.add(name)
                        else:
                            name = "Unknown"
                    except Exception:
                        sims = cosine_similarity(emb.reshape(1, -1), class_means_matrix)[0]
                        best_idx = int(np.argmax(sims))
                        best_sim = sims[best_idx]
                        cos_dist = 1 - best_sim
                        is_known = cos_dist <= COSINE_THRESHOLD
                        if is_known:
                            name = label_map[int(unique_labels[best_idx])]
                            people.add(name)
                        else:
                            name = "Unknown"

                elif model_type == "oneclass":
                    try:
                        pred = model.predict(emb.reshape(1, -1))[0]
                        if pred == 1:
                            u = np.unique(labels_db)
                            if len(u) == 1:
                                name = label_map[int(u[0])]
                                people.add(name)
                            else:
                                name = "Known"
                            is_known = True
                        else:
                            name = "Unknown"
                            is_known = False
                    except Exception:
                        sims = cosine_similarity(emb.reshape(1, -1), class_means_matrix)[0]
                        best_idx = int(np.argmax(sims))
                        best_sim = sims[best_idx]
                        cos_dist = 1 - best_sim
                        is_known = cos_dist <= COSINE_THRESHOLD
                        if is_known:
                            name = label_map[int(unique_labels[best_idx])]
                            people.add(name)
                        else:
                            name = "Unknown"
                else:
                    sims = cosine_similarity(emb.reshape(1, -1), class_means_matrix)[0]
                    best_idx = int(np.argmax(sims))
                    best_sim = sims[best_idx]
                    cos_dist = 1 - best_sim
                    is_known = cos_dist <= COSINE_THRESHOLD
                    if is_known:
                        name = label_map[int(unique_labels[best_idx])]
                        people.add(name)
                    else:
                        name = "Unknown"

                # emotion (cached)
                track_key = f"{name}_{x1}_{y1}_{x2}_{y2}"
                if track_key not in tracks:
                    tracks[track_key] = FaceTrack(name)
                tr = tracks[track_key]

                if (frame_id - tr.last_emo_frame) >= EMO_EVERY_N:
                    try:
                        emo_res = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')
                        if isinstance(emo_res, list):
                            dominant = emo_res[0].get('dominant_emotion', "N/A")
                        else:
                            dominant = emo_res.get('dominant_emotion', "N/A")
                        tr.emotion = dominant
                    except Exception:
                        tr.emotion = "N/A"
                    tr.last_emo_frame = frame_id

                color = (0,255,0) if is_known else (0,0,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                txt = f"{name} | {tr.emotion}"
                cv2.putText(frame, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # fps overlay
        t1 = time.time()
        fps = 1 / (t1 - t0) if (t1 - t0) > 0 else 0.0
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        t0 = t1
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # encode as JPEG
        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()

        # yield multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_attendance_data')
def get_attendance_data():
    """Return all students with their attendance status based on detected faces"""
    try:
        attendance_list = []
        
        # Check if students_data exists and is loaded
        if not students_data:
            return jsonify({"error": "No student data available"}), 500
        
        for student in students_data:
            student_name = student.get('name', '')
            # Check if the student's name is in the detected people set
            status = 'present' if student_name in people else 'absent'
            
            attendance_list.append({
                'name': student_name,
                'rollNo': student.get('roll_number', 'N/A'),
                'std': student.get('standard', 'N/A'),
                'div': student.get('div', 'N/A'),  # Note: 'div' not 'division'
                'status': status
            })
        
        return jsonify(attendance_list), 200
        
    except Exception as e:
        print(f"Error in get_attendance_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    # set host to 0.0.0.0 if you want other devices on your network to access it
    app.run(host='0.0.0.0', port=5000, threaded=True)
