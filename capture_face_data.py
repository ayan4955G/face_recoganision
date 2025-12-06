import cv2
import os
from facenet_pytorch import MTCNN
import pathlib as path
import json

mtcnn = MTCNN(keep_all=False, device='cpu')

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "face_dataset") 

person_name = input("Enter the name of the person: ")
roll_number = int(input("Enter roll number :")) 
std = int(input("Enter standard (it should be between):")) 
div = input("Enter division (it should be alphabet):")

data = {"name": person_name, "roll_number": roll_number, "std": std, "div": div}


save_path = os.path.join(dataset_path, person_name)

print(save_path)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
max_images = 200  # <-- STOP after saving 200 images

print("\nStarting face capture... Will stop at 200 images OR when you press 'q'.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # crop the face
            face = frame[y1:y2, x1:x2]

            if face.size != 0:
                face = cv2.resize(face, (150, 150))
                file_path = os.path.join(save_path, f"{count}.jpg")
                cv2.imwrite(file_path, face)
                count += 1

            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"Captured: {count}/200", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("MTCNN Face Capture", frame)

    # -------- CHECK EXIT CONDITIONS -------- #
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q') or key == 27:
        print("\nStopped manually by user.\n")
        break

    if count >= max_images:
        print("\nReached 200 images. Auto-stopping...\n")
        break
    # -------------------------------------- #

students_file = "students.json"

# Load existing students data
try:
    with open(students_file, "r") as f:
        students_list = json.load(f)
    # Handle case where file contains a single dict instead of a list
    if isinstance(students_list, dict):
        students_list = [students_list]
except FileNotFoundError:
    students_list = []
except json.JSONDecodeError:
    print("Warning: students.json is corrupted. Starting fresh.")
    students_list = []

# Check if student already exists (by name)
existing_student = None
for i, student in enumerate(students_list):
    if student.get('name') == person_name:
        existing_student = i
        break

# Update existing or add new
if existing_student is not None:
    print(f"Updating existing student: {person_name}")
    students_list[existing_student] = data
else:
    print(f"Adding new student: {person_name}")
    students_list.append(data)

# Save back to file
with open(students_file, "w") as f:
    json.dump(students_list, f, indent=4)

print(f"Student data saved to {students_file}")


cap.release()
cv2.destroyAllWindows()


print(f"\nSaved {count} images in '{save_path}'")
