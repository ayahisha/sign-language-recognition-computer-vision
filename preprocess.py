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



 
def augment_data(image):
    # Random rotation
    angle = np.random.uniform(-20, 20)
    M_rotate = cv2.getRotationMatrix2D((24, 24), angle, 1)
    image = cv2.warpAffine(image, M_rotate, (48, 48))

    # Random translation
    M_translate = np.float32([[1, 0, np.random.uniform(-5, 5)], [0, 1, np.random.uniform(-5, 5)]])
    image = cv2.warpAffine(image, M_translate, (48, 48))

    # Random zoom
    zoom_factor = np.random.uniform(0.9, 1.1)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M_zoom = cv2.getRotationMatrix2D(center, 0, zoom_factor)
    image = cv2.warpAffine(image, M_zoom, (w, h))

    #  Gaussian noise
    noise = np.random.normal(0, 10, image.shape).astype('uint8')
    image = cv2.add(image, noise)

    # Random horizontal flip
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)

    return image
