import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import messagebox
import threading

# Load models
gender_model = load_model('./Gender_Classification/Gender-Detection/gender_model.h5')
emotion_model = load_model('./model/fer2013_cnn_improved.h5')

gender_labels = ['female', 'male']
emotion_labels = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprise']

# MediaPipe face detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Tkinter window setup
root = tk.Tk()
root.geometry('1050x620')
root.resizable(False, False)
root.title('VKU')

icon_img = Image.open('./img/VKU.png')
icon = ImageTk.PhotoImage(icon_img)
root.iconphoto(True, icon)

is_running = False

def use_camera():
    global is_running
    is_running = True
    start_button.config(state="disabled")
    stop_button.config(state="normal")
    exit_button.config(state="normal")
    threading.Thread(target=camera_worker, daemon=True).start()

def quit_program():
    if messagebox.askyesno("Quit", "Do you want to exit?"):
        root.destroy()

def cancel_feed():
    global is_running
    is_running = False
    start_button.config(state="normal")
    stop_button.config(state="disabled")

def camera_worker():
    capture = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while is_running:
            ret, frame = capture.read()
            if not ret:
                break

            # Convert the BGR image to RGB before processing.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    # Convert relative bbox to absolute pixel coordinates
                    x1 = int(bboxC.xmin * iw)
                    y1 = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)

                    # Ensure coordinates are within frame bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(iw, x1 + w)
                    y2 = min(ih, y1 + h)

                    # Draw rectangle on face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Crop face region
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                        continue

                    # Prepare input for gender model (grayscale 48x48)
                    face_gray_for_gender = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    face_gray_for_gender = cv2.resize(face_gray_for_gender, (48, 48))
                    face_gray_for_gender = face_gray_for_gender.astype("float32") / 255.0
                    face_gray_for_gender = np.expand_dims(face_gray_for_gender, axis=-1)  # (48,48,1)
                    face_gray_for_gender = np.expand_dims(face_gray_for_gender, axis=0)   # (1,48,48,1)

                    # Prepare input for emotion model (grayscale 48x48)
                    face_gray_for_emotion = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    face_gray_for_emotion = cv2.resize(face_gray_for_emotion, (48, 48))
                    face_gray_for_emotion = face_gray_for_emotion.astype("float32") / 255.0
                    face_gray_for_emotion = np.expand_dims(face_gray_for_emotion, axis=-1)
                    face_gray_for_emotion = np.expand_dims(face_gray_for_emotion, axis=0)  # (1,48,48,1)

                    # Predict gender
                    gender_pred = gender_model.predict(face_gray_for_gender, verbose=0)[0]
                    gender = gender_labels[np.argmax(gender_pred)]

                    # Predict emotion
                    emotion_pred = emotion_model.predict(face_gray_for_emotion, verbose=0)[0]
                    emotion = emotion_labels[np.argmax(emotion_pred)]

                    label = f"{gender}, {emotion}"
                    Y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                    cv2.putText(frame, label, (x1, Y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Show image in Tkinter
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = image.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=image)
            image_label.configure(image=imgtk)
            image_label.image = imgtk

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

    capture.release()
    cv2.destroyAllWindows()

# UI layout

# Main frame
main_frame = tk.Frame(root, bg='#f0e0c2')
main_frame.pack(fill=tk.BOTH, expand=True)
main_frame.pack_propagate(False)
main_frame.configure(width=1050, height=620)

# Tiêu đề
label_title = tk.Label(main_frame, text='PREDICT GENDER AND EMOTION BASED ON HUMAN FACES',
                       font=("Arial", 20), fg="red", bg='#f0e0c2')
label_title2 = tk.Label(main_frame, text='MyDang, DaoKhuyen',
                        font=("Arial", 15), fg="green", bg='#f0e0c2')

# Label hiển thị camera
image_label = tk.Label(main_frame, bg='#f0e0c2')
image_label.place(x=160, y=110, width=750, height=450)

# Nút EXIT
def quit_program():
    if messagebox.askyesno("Quit", "Do you want to exit?"):
        root.destroy()

# Nút START
start_button = tk.Button(main_frame, text="START", font=('Bold', 15), fg='white', bd=0,
                         bg='green', command=use_camera)
start_button.place(x=300, y=570, width=80, height=35)

# Nút STOP
stop_button = tk.Button(main_frame, text="STOP", font=('Bold', 15), fg='white', bd=0,
                        bg='green', command=cancel_feed, state="disabled")
stop_button.place(x=500, y=570, width=80, height=35)

# Nút EXIT
exit_button = tk.Button(main_frame, text="EXIT", font=('Bold', 15), fg='white', bd=0,
                        bg='green', command=quit_program, state="normal")
exit_button.place(x=700, y=570, width=80, height=35)

label_title.pack()
label_title2.pack()

root.mainloop()
