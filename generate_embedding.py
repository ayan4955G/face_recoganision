import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "face_dataset") 

device = 'cpu'

mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []
labels = []
label_map = {}

current_id = 0

print("\nGenerating embeddings...\n")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder):
        continue

    print(f"Processing: {person_name}")
    label_map[current_id] = person_name

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # MTCNN face detection
        face = mtcnn(img_rgb)
        if face is None:
            continue

        # Generate embedding
        face_emb = resnet(face.unsqueeze(0)).detach().numpy()
        embeddings.append(face_emb)
        labels.append(current_id)

    current_id += 1

embeddings = np.array(embeddings).reshape(len(embeddings), 512)
labels = np.array(labels)

save_dir = os.path.dirname(os.path.realpath(__file__))

np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
np.save(os.path.join(save_dir, "embedding_labels.npy"), labels)
np.save(os.path.join(save_dir, "label_map.npy"), label_map)


print("\nEmbeddings generated successfully!")
print("Total embeddings:", len(embeddings))
