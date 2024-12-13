from train_model import train_classifier_letters, train_classifier_numbers
from recognize import run_recognition_letters, run_recognition_numbers

if __name__ == "__main__":
    print("Options:\n1. Train Letters Classifier\n2. Train Numbers Classifier\n3. Run Letter Recognition\n4. Run Number Recognition")
    choice = input("Enter your choice: ")

    if choice == '1':
        train_classifier_letters()
    elif choice == '2':
        train_classifier_numbers()
    elif choice == '3':
        run_recognition_letters()
    elif choice == '4':
        run_recognition_numbers()
    else:
        print("Invalid choice.")
