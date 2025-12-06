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

# ---------- CONFIG ----------
RTSP_URL = "rtsp://admin:L2E9B7FC@10.47.29.210:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MIN_FACE_CONF = 0.95        # optional threshold for detection confidence
COSINE_THRESHOLD = 0.45
SVM_PROB_THRESHOLD = 0.50
EMO_EVERY_N = 10            # compute emotion every N frames per face

# ---------- LOAD MODELS ----------
print("Loading models...")

# Resnet for embeddings
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# MTCNN face detector (PyTorch)
mtcnn = MTCNN(keep_all=False, device=DEVICE if DEVICE == 'cuda' else 'cpu')

# Load DB files (paths relative to script)
base_dir = os.path.dirname(os.path.realpath(__file__))
emb_file = os.path.join(base_dir, "embeddings.npy")
labels_file = os.path.join(base_dir, "embedding_labels.npy")
label_map_file = os.path.join(base_dir, "label_map.npy")

# classifier candidates
svm_path = os.path.join(base_dir, "svm_model.joblib")
oneclass_path = os.path.join(base_dir, "oneclass_model.joblib")

# load embeddings / labels
if not os.path.exists(emb_file) or not os.path.exists(labels_file) or not os.path.exists(label_map_file):
    raise FileNotFoundError("embeddings.npy, embedding_labels.npy or label_map.npy missing in model folder")

embeddings_db = np.load(emb_file)
labels_db = np.load(labels_file)
label_map = np.load(label_map_file, allow_pickle=True).item()

# Normalize DB embeddings and compute class means
embeddings_db = normalize(embeddings_db)
unique_labels = np.unique(labels_db)

if len(unique_labels) == 0:
    raise RuntimeError("No labels found in label DB.")

class_means_matrix = np.vstack([np.mean(embeddings_db[labels_db == u], axis=0)
                                for u in unique_labels])
class_order = list(unique_labels)

# Load classifier if exists (SVC or OneClassSVM)
model = None
model_type = "none"  # "svm", "oneclass", "none"

if os.path.exists(svm_path):
    model = joblib.load(svm_path)
    model_type = "svm"
    print("Loaded multi-class SVM:", svm_path)
elif os.path.exists(oneclass_path):
    model = joblib.load(oneclass_path)
    model_type = "oneclass"
    print("Loaded OneClassSVM:", oneclass_path)
else:
    print("No classifier file found (svm_model.joblib or oneclass_model.joblib). Will use cosine-only matching.")

print("Models loaded.")

# simple cache for emotion & identity
class FaceTrack:
    def __init__(self, name):
        self.name = name
        self.last_emo_frame = -999
        self.emotion = "N/A"

tracks = {}

# ---------- OPEN CAMERA / RTSP ----------
# cap = cv2.VideoCapture(RTSP_URL)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open video stream")

print("Running...")

frame_id = 0
t0 = time.time()
fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    frame_id += 1
    h, w = frame.shape[:2]

    # MTCNN detection: returns boxes (N,4) and probs (N,)
    boxes, probs = mtcnn.detect(frame)

    if boxes is None:
        cv2.imshow("Smooth CCTV", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # iterate detections
    for i, box in enumerate(boxes):
        prob = probs[i] if probs is not None else 1.0
        # optional threshold
        if prob < MIN_FACE_CONF:
            continue

        x1, y1, x2, y2 = map(int, box)
        # clamp coordinates
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
            emb = resnet(t).cpu().numpy()   # shape (1,512)

        # normalize embedding (2D)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # shape (1,512)

        # ------ Identification: handle SVC / OneClassSVM / cosine fallback ------
        name = "Unknown"
        is_known = False

        if model_type == "svm":
            # multi-class SVC with predict_proba
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
                else:
                    name = "Unknown"
            except Exception:
                # fallback to cosine-only
                sims = cosine_similarity(emb.reshape(1, -1), class_means_matrix)[0]
                best_idx = int(np.argmax(sims))
                best_sim = sims[best_idx]
                cos_dist = 1 - best_sim
                is_known = cos_dist <= COSINE_THRESHOLD
                if is_known:
                    name = label_map[int(unique_labels[best_idx])]
                else:
                    name = "Unknown"

        elif model_type == "oneclass":
            # OneClassSVM: predict -> +1 means inlier (known), -1 outlier (unknown)
            try:
                pred = model.predict(emb.reshape(1, -1))[0]
                if pred == 1:
                    # if label_map contains a single label, map to that; otherwise generic Known
                    u = np.unique(labels_db)
                    if len(u) == 1:
                        name = label_map[int(u[0])]
                    else:
                        name = "Known"
                    is_known = True
                else:
                    name = "Unknown"
                    is_known = False
            except Exception:
                # fallback to cosine
                sims = cosine_similarity(emb.reshape(1, -1), class_means_matrix)[0]
                best_idx = int(np.argmax(sims))
                best_sim = sims[best_idx]
                cos_dist = 1 - best_sim
                is_known = cos_dist <= COSINE_THRESHOLD
                if is_known:
                    name = label_map[int(unique_labels[best_idx])]
                else:
                    name = "Unknown"

        else:
            # cosine-only fallback
            sims = cosine_similarity(emb.reshape(1, -1), class_means_matrix)[0]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]
            cos_dist = 1 - best_sim
            is_known = cos_dist <= COSINE_THRESHOLD
            if is_known:
                name = label_map[int(unique_labels[best_idx])]
            else:
                name = "Unknown"

        # ----- Emotion (cached) -----
        track_key = f"{name}_{x1}_{y1}_{x2}_{y2}"
        if track_key not in tracks:
            tracks[track_key] = FaceTrack(name)
        tr = tracks[track_key]

        if (frame_id - tr.last_emo_frame) >= EMO_EVERY_N:
            try:
                # use mtcnn backend to avoid TensorFlow retinaface dependency inside DeepFace
                emo_res = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')
                if isinstance(emo_res, list):
                    dominant = emo_res[0].get('dominant_emotion', "N/A")
                else:
                    dominant = emo_res.get('dominant_emotion', "N/A")
                tr.emotion = dominant
            except Exception:
                tr.emotion = "N/A"
            tr.last_emo_frame = frame_id

        # ----- Draw -----
        color = (0,255,0) if is_known else (0,0,255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        txt = f"{name} | {tr.emotion}"
        cv2.putText(frame, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255,255,255), 2)

    # FPS
    t1 = time.time()
    fps = 1 / (t1 - t0) if (t1 - t0) > 0 else 0.0
    fps_list.append(fps)
    if len(fps_list) > 30:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)
    t0 = t1

    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Smooth CCTV", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
