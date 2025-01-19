import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from preprocess import extract_landmarks
import mediapipe as mp

DATA_DIR = './data'
LETTER_DIR = os.path.join(DATA_DIR, 'letters')
NUMBER_DIR = os.path.join(DATA_DIR, 'numbers')
MODEL_FILE_LETTERS = './sign_recognition_model_knn_letters.pkl'
MODEL_FILE_NUMBERS = './sign_recognition_model_knn_numbers.pkl'

mp_hands = mp.solutions.hands

def train_classifier(data_dir, model_file, data_type):
    print(f"Training for {data_type}...")
    X, y = [], []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
        for gesture_class in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, gesture_class)
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path)
                image = cv2.GaussianBlur(image, (5, 5), 0)
                landmarks, _, _ = extract_landmarks(image, hands)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(gesture_class)

    if not X:
        print(f"No data found for {data_type}. Please collect data first.")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"Splitting dataset for {data_type}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training classifier for {data_type}...")
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training completed for {data_type} with accuracy: {accuracy * 100:.2f}%")
    
   
    print(f"Classification Report for {data_type}:\n", classification_report(
        y_test, y_pred, target_names=np.unique(y_test)
    ))
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {data_type}:\n", cm)

    joblib.dump(classifier, model_file)
    print(f"{data_type.capitalize()} recognition model saved!")

def train_classifier_letters():
    train_classifier(LETTER_DIR, MODEL_FILE_LETTERS, "Letters (A-Z)")

def train_classifier_numbers():
    train_classifier(NUMBER_DIR, MODEL_FILE_NUMBERS, "Numbers (0-9)")
