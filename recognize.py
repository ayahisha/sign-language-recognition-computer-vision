import cv2
import joblib
import mediapipe as mp
from preprocess import extract_landmarks
import os

# Paths and settings
MODEL_FILE_LETTERS = './gesture_recognition_model_knn_letters.pkl'
MODEL_FILE_NUMBERS = './gesture_recognition_model_knn_numbers.pkl'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
def load_model(model_file):
    if os.path.exists(model_file):
        return joblib.load(model_file)
    print(f"Model not found at {model_file}. Please train the model first.")
    return None

def run_recognition_letters():
    classifier = load_model(MODEL_FILE_LETTERS)
    if classifier is None:
        return

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            landmarks, frame_with_landmarks, hand_landmarks = extract_landmarks(frame, hands)
            if landmarks is not None:
                predicted_label = classifier.predict([landmarks])[0]

                if hand_landmarks:
                    for landmarks in hand_landmarks:
                        mp_drawing.draw_landmarks(frame_with_landmarks, landmarks, mp_hands.HAND_CONNECTIONS)

                x_coords = [lm.x * w for lm in hand_landmarks[0].landmark]
                y_coords = [lm.y * h for lm in hand_landmarks[0].landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cv2.rectangle(frame_with_landmarks, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame_with_landmarks, predicted_label, (int(x_min) + 10, int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Letter Sign Language Recognition", frame_with_landmarks)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def run_recognition_numbers():
    classifier = load_model(MODEL_FILE_NUMBERS)
    if classifier is None:
        return

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            landmarks, frame_with_landmarks, hand_landmarks = extract_landmarks(frame, hands)
            if landmarks is not None:
                predicted_label = classifier.predict([landmarks])[0]

                if hand_landmarks:
                    for landmarks in hand_landmarks:
                        mp_drawing.draw_landmarks(frame_with_landmarks, landmarks, mp_hands.HAND_CONNECTIONS)

                x_coords = [lm.x * w for lm in hand_landmarks[0].landmark]
                y_coords = [lm.y * h for lm in hand_landmarks[0].landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cv2.rectangle(frame_with_landmarks, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame_with_landmarks, predicted_label, (int(x_min) + 10, int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Number Sign Language Recognition", frame_with_landmarks)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
