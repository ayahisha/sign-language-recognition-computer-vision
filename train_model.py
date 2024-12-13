import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
from preprocess import extract_landmarks
import mediapipe as mp

# Paths and settings
DATA_DIR = './data'
LETTER_DIR = os.path.join(DATA_DIR, 'letters')
NUMBER_DIR = os.path.join(DATA_DIR, 'numbers')
MODEL_FILE_LETTERS = './gesture_recognition_model_knn_letters.pkl'
MODEL_FILE_NUMBERS = './gesture_recognition_model_knn_numbers.pkl'

mp_hands = mp.solutions.hands

def train_classifier_letters():
    print("Training for Letters (A-Z)...")
    X, y = [], []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
        for gesture_class in os.listdir(LETTER_DIR):
            class_dir = os.path.join(LETTER_DIR, gesture_class)
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path)
                image = cv2.GaussianBlur(image, (5, 5), 0)
                landmarks, _, _ = extract_landmarks(image, hands)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(gesture_class)

    if not X:
        print("No data found for letters. Please collect data first.")
        return

    X = np.array(X)
    y = np.array(y)

    print("Splitting dataset for letters...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training classifier for letters...")
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training completed for letters with accuracy: {accuracy * 100:.2f}%")

    print("Classification Report for letters:\n", classification_report(
        y_test, y_pred, labels=np.unique(y_test), target_names=[str(label) for label in np.unique(y_test)]
    ))

    joblib.dump(classifier, MODEL_FILE_LETTERS)
    print("Letter recognition model saved!")

def train_classifier_numbers():
    print("Training for Numbers (0-9)...")
    X, y = [], []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
        for gesture_class in os.listdir(NUMBER_DIR):
            class_dir = os.path.join(NUMBER_DIR, gesture_class)
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path)
                image = cv2.GaussianBlur(image, (5, 5), 0)
                landmarks, _, _ = extract_landmarks(image, hands)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(gesture_class)

    if not X:
        print("No data found for numbers. Please collect data first.")
        return

    X = np.array(X)
    y = np.array(y)

    print("Splitting dataset for numbers...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training classifier for numbers...")
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training completed for numbers with accuracy: {accuracy * 100:.2f}%")

    print("Classification Report for numbers:\n", classification_report(
        y_test, y_pred, labels=np.unique(y_test), target_names=[str(label) for label in np.unique(y_test)]
    ))

    joblib.dump(classifier, MODEL_FILE_NUMBERS)
    print("Number recognition model saved!")
