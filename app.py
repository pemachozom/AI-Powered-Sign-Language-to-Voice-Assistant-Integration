from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Load the Keras model
model = load_model('gesture.keras')

# Define class labels
actions = ["dangerous", "kuzuzangpo", "washroom"]
class_labels = np.array(actions)

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def extract_landmarks_from_frame(frame):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        landmarks = extract_keypoints(results)
        return landmarks

def preprocess_landmarks(landmarks_list):
    sequence_length = 30
    window = []
    if landmarks_list:
        for landmarks_frame in landmarks_list:
            window.append(landmarks_frame)
        while len(window) < sequence_length:
            window.append(np.zeros_like(landmarks_list[0]))
    else:
        for _ in range(sequence_length):
            window.append(np.zeros(1662,))
    window_array = np.array(window)
    return window_array

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // 30, 1)
    landmarks_list = []
    frame_counter = 0
    while cap.isOpened() and len(landmarks_list) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_counter % frame_interval == 0:
            landmarks = extract_landmarks_from_frame(frame)
            landmarks_list.append(landmarks)
        frame_counter += 1
    cap.release()
    return preprocess_landmarks(landmarks_list)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        file_path = 'uploaded_video_webcam.mp4'
        file.save(file_path)
        landmarks_p = process_video(file_path)
        landmarks_p = landmarks_p.reshape((1, 30, 1662))
        prediction = model.predict(landmarks_p)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        return jsonify({'predicted_class': predicted_class_label})
    else:
        return jsonify({'error': 'No file uploaded'}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
