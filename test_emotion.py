import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load mô hình đã lưu
model_path = "./model/fer2013_cnn_improved.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

model = load_model(model_path)

# Đường dẫn ảnh
img_path = "manyFace.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

# Load bộ phát hiện khuôn mặt (Haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load và nhận diện nhiều khuôn mặt
def predict_multiple_faces(img_path, model):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("Không tìm thấy khuôn mặt nào trong ảnh.")
        return

    emotion_labels = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprise']

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0  # Chuẩn hóa
        face = np.reshape(face, (1, 48, 48, 1))

        # Dự đoán cảm xúc
        prediction = model.predict(face)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        # Vẽ khung + nhãn cảm xúc lên ảnh gốc
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Hiển thị ảnh
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

# Gọi hàm
predict_multiple_faces(img_path, model)
