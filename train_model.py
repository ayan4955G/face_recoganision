# train_model.py
import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, OneClassSVM
import joblib

base_dir = os.path.dirname(os.path.realpath(__file__))

emb_path = os.path.join(base_dir, "embeddings.npy")
labels_path = os.path.join(base_dir, "embedding_labels.npy")

emb = np.load(emb_path)
labels = np.load(labels_path)

emb = normalize(emb)
unique, counts = np.unique(labels, return_counts=True)
print("Base dir:", base_dir)
print("Unique labels:", unique)
print("Counts per label:", counts)

if len(unique) > 1:
    print("Training multi-class SVM...")
    svm = SVC(probability=True, kernel='linear')
    svm.fit(emb, labels)
    save_path = os.path.join(base_dir, "svm_model.joblib")
    joblib.dump(svm, save_path)
    print("Saved multi-class SVM to", save_path)
else:
    print("Only one class found. Training OneClassSVM (novelty detector)...")
    oc = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)  # tune nu as needed
    oc.fit(emb)
    save_path = os.path.join(base_dir, "oneclass_model.joblib")
    joblib.dump(oc, save_path)
    print("Saved OneClassSVM to", save_path)

print("Done.")
