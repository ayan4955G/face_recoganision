# detector.py
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

class FaceTrack:
    def __init__(self, name):
        self.name = name
        self.last_emo_frame = -999
        self.emotion = "N/A"

class FaceDetector:
    def __init__(self,
                 model_dir='.',
                 device=None,
                 min_face_conf=0.95,
                 cosine_threshold=0.45,
                 svm_prob_threshold=0.5,
                 emo_every_n=10):
        """
        model_dir: directory containing embeddings.npy, embedding_labels.npy, label_map.npy, (optional) svm_model.joblib
        """
        self.model_dir = os.path.abspath(model_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.MIN_FACE_CONF = min_face_conf
        self.COSINE_THRESHOLD = cosine_threshold
        self.SVM_PROB_THRESHOLD = svm_prob_threshold
        self.EMO_EVERY_N = emo_every_n

        # caches
        self.tracks = {}

        # load models
        self._load_models()

    def _load_models(self):
        print("[detector] loading models on", self.device)
        # Resnet
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # MTCNN: keep_all False (we detect single face per crop) — we will call detect on full frame
        self.mtcnn = MTCNN(keep_all=False, device=self.device if self.device == 'cuda' else 'cpu')

        emb_file = os.path.join(self.model_dir, "embeddings.npy")
        labels_file = os.path.join(self.model_dir, "embedding_labels.npy")
        label_map_file = os.path.join(self.model_dir, "label_map.npy")
        svm_path = os.path.join(self.model_dir, "svm_model.joblib")
        oneclass_path = os.path.join(self.model_dir, "oneclass_model.joblib")

        if not (os.path.exists(emb_file) and os.path.exists(labels_file) and os.path.exists(label_map_file)):
            raise FileNotFoundError("embeddings.npy, embedding_labels.npy or label_map.npy missing in model folder")

        self.embeddings_db = np.load(emb_file)
        self.labels_db = np.load(labels_file)
        self.label_map = np.load(label_map_file, allow_pickle=True).item()
        self.embeddings_db = normalize(self.embeddings_db)
        self.unique_labels = np.unique(self.labels_db)
        if len(self.unique_labels) == 0:
            raise RuntimeError("No labels found in label DB.")
        self.class_means_matrix = np.vstack([np.mean(self.embeddings_db[self.labels_db == u], axis=0)
                                            for u in self.unique_labels])

        self.model = None
        self.model_type = "none"
        if os.path.exists(svm_path):
            self.model = joblib.load(svm_path)
            self.model_type = "svm"
            print("[detector] loaded SVM:", svm_path)
        elif os.path.exists(oneclass_path):
            self.model = joblib.load(oneclass_path)
            self.model_type = "oneclass"
            print("[detector] loaded oneclass:", oneclass_path)
        else:
            print("[detector] no classifier; using cosine-only")

        print("[detector] models loaded")

    def _get_embedding(self, face):
        """face: BGR numpy image of face ROI. returns normalized (1,512) numpy embedding or None."""
        try:
            face_resized = cv2.resize(face, (160, 160))
            rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            t = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            t = (t - 127.5) / 128.0
            with torch.no_grad():
                emb = self.resnet(t).cpu().numpy()
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            return emb
        except Exception as e:
            print("[detector] embedding error:", e)
            return None

    def _identify(self, emb):
        """emb shape (1,512) normalized. returns (name, is_known)"""
        if emb is None:
            return "Unknown", False

        if self.model_type == "svm":
            try:
                probs_out = self.model.predict_proba(emb)[0]
                pred_idx = int(np.argmax(probs_out))
                svc_prob = probs_out[pred_idx]
                pred_label = self.model.classes_[pred_idx]

                sims = cosine_similarity(emb.reshape(1, -1), self.class_means_matrix)[0]
                best_idx = int(np.argmax(sims))
                cos_dist = 1 - sims[best_idx]

                is_known = (svc_prob >= self.SVM_PROB_THRESHOLD) and (cos_dist <= self.COSINE_THRESHOLD)
                if is_known:
                    return self.label_map[int(pred_label)], True
                else:
                    return "Unknown", False
            except Exception:
                pass  # fallback to cosine-only below

        # oneclass fallback handled similarly or cosine-only
        sims = cosine_similarity(emb.reshape(1, -1), self.class_means_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        cos_dist = 1 - best_sim
        is_known = cos_dist <= self.COSINE_THRESHOLD
        if is_known:
            return self.label_map[int(self.unique_labels[best_idx])], True
        else:
            return "Unknown", False

    def process_frame(self, frame, frame_id=0):
        """
        Annotate and return the frame (BGR numpy) with rectangles + text.
        frame_id: increasing integer to control emotion frequency caching.
        """
        if frame is None:
            return None

        h, w = frame.shape[:2]
        boxes, probs = self.mtcnn.detect(frame)

        if boxes is None:
            return frame

        for i, box in enumerate(boxes):
            prob = probs[i] if probs is not None else 1.0
            if prob < self.MIN_FACE_CONF:
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            emb = self._get_embedding(face)
            name, is_known = self._identify(emb)

            # emotion (cached per bounding-box + name)
            track_key = f"{name}_{x1}_{y1}_{x2}_{y2}"
            if track_key not in self.tracks:
                self.tracks[track_key] = FaceTrack(name)
            tr = self.tracks[track_key]

            if (frame_id - tr.last_emo_frame) >= self.EMO_EVERY_N:
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

        return frame
