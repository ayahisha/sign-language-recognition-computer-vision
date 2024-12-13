import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

def extract_landmarks(image, hands):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    if results.multi_hand_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
        return normalize_landmarks(landmarks).flatten(), image, results.multi_hand_landmarks
    return None, image, None

def normalize_landmarks(landmarks):
    min_vals = np.min(landmarks, axis=0)
    max_vals = np.max(landmarks, axis=0)
    return (landmarks - min_vals) / (max_vals - min_vals)
