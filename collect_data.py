import os
import cv2

# Paths and settings
DATA_DIR = './data'
LETTER_DIR = os.path.join(DATA_DIR, 'letters')
NUMBER_DIR = os.path.join(DATA_DIR, 'numbers')
CLASSES_LETTERS = [chr(i) for i in range(65, 91)]  # A to Z
CLASSES_NUMBERS = [str(i) for i in range(10)]  # 0 to 9
IMAGES_PER_CLASS = 100  # Number of images per class

def collect_data():
    cap = cv2.VideoCapture(0)
    os.makedirs(LETTER_DIR, exist_ok=True)
    os.makedirs(NUMBER_DIR, exist_ok=True)

    print("Options:\n1. Collect Letters\n2. Collect Numbers")
    choice = input("Enter your choice: ")

    if choice == '1':
        classes = CLASSES_LETTERS
        parent_dir = LETTER_DIR
        print("Collecting data for letters (A-Z).")
    elif choice == '2':
        classes = CLASSES_NUMBERS
        parent_dir = NUMBER_DIR
        print("Collecting data for numbers (0-9).")
    else:
        print("Invalid choice.")
        return

    for gesture_class in classes:
        class_dir = os.path.join(parent_dir, gesture_class)
        os.makedirs(class_dir, exist_ok=True)

        print(f"Collecting data for class '{gesture_class}'. Press 'q' to start.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame!")
                return
            cv2.putText(frame, f"Press 'q' to start capturing {gesture_class}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Collecting Data', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"Capturing images for class '{gesture_class}'...")
        for i in range(IMAGES_PER_CLASS):
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame!")
                return
            frame = cv2.flip(frame, 1)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            cv2.imshow('Collecting Data', frame)
            cv2.imwrite(os.path.join(class_dir, f'{i}.jpg'), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
